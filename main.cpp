#include <X11/X.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cpr/cpr.h>

#include "cpr/cprtypes.h"
#include "cpr/response.h"
#include "xdp_stream.hpp"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <optional>
#include <tesseract/baseapi.h>

#include <tesseract/publictypes.h>
#include <thread>
#include <uiohook.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <fmt/format.h>

#include <unordered_set>
#include <vector>
#include "json.hpp"

constexpr bool save_frames_to_disk = true;
template<typename mat>
void save_image(std::string path, mat m) {
    if constexpr (save_frames_to_disk) {
        cv::imwrite(path, m);
    }
}

static bool record_frame = false;
const std::string base_path = "/home/gudov/src/warfeye";

struct ProcessedImgs {
    cv::Mat counters;
    cv::Mat masked;
};

ProcessedImgs processFrame(cv::Mat &&img) {
    ProcessedImgs imgs;
    cv::Mat filtered_small;
    int range_small = 224; // 128
    int threshold_small = 224; // 224
    cv::threshold(img, filtered_small, range_small, 256, 0);
    cv::inRange(filtered_small, cv::Scalar(threshold_small,threshold_small,threshold_small), cv::Scalar(256,256,256), filtered_small);

    cv::Mat filtered_big;
    cv::inRange(img, cv::Scalar(224,224,224), cv::Scalar(256,256,256), filtered_big);
    cv::blur(filtered_big, filtered_big, cv::Size(3,3));
    cv::threshold(filtered_big, filtered_big, 48, 256, 0);
    cv::blur(filtered_big, filtered_big, cv::Size(5,5));
    cv::threshold(filtered_big, filtered_big, 32, 256, 0);

    save_image("filtered_small.png", filtered_small);
    save_image("filtered_big.png", filtered_big);
    cv::copyTo(filtered_small, imgs.masked, filtered_big);
    cv::threshold(imgs.masked, imgs.masked, 128, 256, 1);
    save_image("masked.png", imgs.masked);

    // cv::blur(filtered_big, counter, cv::Size(3,3));
    // cv::threshold(counter, counter, 1, 256, 0);
    cv::threshold(filtered_big, imgs.counters, 1, 256, 0);
    save_image("counters.png", imgs.counters);

    return imgs;
}

struct Cut {
    cv::Rect pos;
    cv::Mat mat;
};

std::vector<Cut> cutImages(ProcessedImgs &imgs) {
    std::vector<std::vector<cv::Point>> contours;
    findContours(imgs.counters, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Rect> boundRect( contours.size() );

    for( size_t i = 0; i < contours.size(); i++ ) {
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        boundRect[i] = boundingRect( contours_poly[i] );
        int max_w = imgs.counters.cols;
        boundRect[i].width = std::min(boundRect[i].width + boundRect[i].height, max_w);
        int max_h = imgs.counters.rows;
        boundRect[i].height = std::min(int(boundRect[i].height * 1.2f), max_w);
    }
 
    std::list<cv::Rect> merged;
    for (auto &cont : boundRect) {
        merged.push_back(cont);
    }

    bool merged_something = false;
    do {
        merged_something = false;
        for (auto i = merged.begin(); i != merged.end(); i++) {
            for (auto j = i; j != merged.end();) {
                if (i == j) {
                    j++;
                    continue;
                }
                bool x_intersect = (i->x < j->x) ? (i->x + i->width >= j->x) : (j->x + j->width >= i->x);
                bool y_intersect = (i->y < j->y) ? (i->y + i->height >= j->y) : (j->y + j->height >= i->y);
                
                auto k = j;
                j++;
                if (x_intersect && y_intersect) {
                    int max_w = std::max(i->x + i->width, k->x + k->width);
                    int max_h = std::max(i->y + i->height, k->y + k->height);
                    i->x = std::min(i->x, k->x);
                    i->y = std::min(i->y, k->y);
                    i->width = max_w - i->x;
                    i->height = max_h - i->y;

                    merged.erase(k);
                    merged_something = true;
                }
            }
        }
    } while (merged_something);

    for (auto i = merged.begin(); i != merged.end();) {
        auto j = i;
        i++;
        if (j->area() <= 1000) {
            merged.erase(j);
        }
    }

    std::vector<Cut> cuts;

    auto it = merged.begin();
    for( size_t i = 0; i < merged.size(); i++ ) {
        cuts.push_back({
            .pos = *it,
            .mat = imgs.masked(*it).clone()
        });
        it++;
    }

    return cuts;
}

tesseract::TessBaseAPI *initTesseract() {
    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    if (api->Init("/usr/share/tessdata/", "eng", tesseract::OEM_LSTM_ONLY)) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    api->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);

    return api;
}

std::string recognizeCut(Cut &cut, tesseract::TessBaseAPI *api) {
    api->SetImage((uchar*)cut.mat.data, cut.mat.size().width, cut.mat.size().height, cut.mat.channels(), cut.mat.step1());
    api->Recognize(0);

    std::string text;
    tesseract::ResultIterator* ri = api->GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
    if (ri != 0) {
        do {
            const char* word = ri->GetUTF8Text(level);
            if (word) {
                std::string word_str = word;
                std::transform(word_str.begin(), word_str.end(), word_str.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                if (!text.empty()) {
                    text += " ";
                }
                text += word_str;
            }
            delete[] word;
        } while (ri->Next(level));
    }

    return text;
}

struct Orders {
    std::vector<int> sell, buy;
};

Orders get_prices(std::string slug) {
    cpr::Response top_orders = cpr::Get(cpr::Url(fmt::format("https://api.warframe.market/v2/orders/item/{}/top", slug)));
    // FILE *all_items_f = fopen("item_top.json", "wb");
    // fwrite(top_orders.text.data(), top_orders.text.size(), 1, all_items_f);
    // fclose(all_items_f);

    auto json = nlohmann::json::parse(top_orders.text);
    Orders orders;

    for (auto &sell_order : json["data"]["sell"]) {
        orders.sell.push_back(sell_order["platinum"]);
    }

    for (auto &buy_order : json["data"]["buy"]) {
        orders.buy.push_back(buy_order["platinum"]);
    }

    for (size_t i = orders.sell.size(); i < 5; i++) {
        orders.sell.push_back(4096);
    }
    for (size_t i = orders.buy.size(); i < 5; i++) {
        orders.buy.push_back(0);
    }

    std::sort(orders.sell.begin(), orders.sell.end(), [](int a, int b){return a > b;});
    std::sort(orders.buy.begin(), orders.buy.end(), [](int a, int b){return a > b;});
    return orders;
}

struct ItemInfo {
    std::string slug;
    bool vaulted;
};

int main(int argc, char *argv[]) {
    std::thread keyboard([](){
        Display*    dpy     = XOpenDisplay(0);
        Window      root    = DefaultRootWindow(dpy);
        XEvent      ev;

        unsigned int    modifiers       = ControlMask | ShiftMask; // AnyModifier
        int             keycode         = XKeysymToKeycode(dpy,XK_P);
        Window          grab_window     =  root;
        Bool            owner_events    = False;
        int             pointer_mode    = GrabModeAsync;
        int             keyboard_mode   = GrabModeAsync;

        int grab_result = XGrabKey(dpy, keycode, modifiers, grab_window, owner_events, pointer_mode,
                keyboard_mode);

        // unsigned int target_modifers = ControlMask | ShiftMask;

        XSelectInput(dpy, root, KeyPressMask );
        while(true) {
            XNextEvent(dpy, &ev);
            // if ((ev.xkey.state & target_modifers) != target_modifers) {
            //     continue;
            // }
            switch(ev.type)
            {
                case KeyPress:
                    std::cout << "Hot key pressed!" << std::endl;
                    record_frame = true;

                default:
                    break;
            }
        }
    });

    auto tess_api = initTesseract();
    std::unordered_set<std::string> filter = {
        "prime",
        "relic"
    };

    std::unordered_set<std::string> exclude = {
        "[radiant]",
        "[flawless]"
    };

    cpr::Response all_items = cpr::Get(cpr::Url("https://api.warframe.market/v2/items"));

    auto items_json = nlohmann::json::parse(all_items.text);
    auto items_data = items_json["data"];
    std::unordered_map<std::string, ItemInfo> items_slugs;
    for (auto item: items_data) {
        std::string name = item["i18n"]["en"]["name"];
        std::transform(name.begin(), name.end(), name.begin(),
                    [](unsigned char c){ return std::tolower(c); });
        // std::cout << "add item: " << name << std::endl;
        items_slugs[name] = {
            .slug = item["slug"],
            .vaulted = item["vaulted"].is_boolean() ? (bool)item["vaulted"] : false
        };
    }

    init_screencast(argc, argv, [&](void*data,uint32_t size,size_t w,size_t h) {
        if (record_frame) {
            record_frame = false;
            std::cout << "image recieved" << std::endl;
            cv::Mat image(h,w, CV_8UC4, (uint8_t*)data);
            cv::Mat image_rgb;
            cv::cvtColor(image, image_rgb, cv::COLOR_RGBA2RGB);
            auto processed = processFrame(std::move(image_rgb));
            auto cuts = cutImages(processed);

            size_t i = 1;
            for (auto &cut: cuts) {
                auto text = recognizeCut(cut, tess_api);
                // std::cout << fmt::format("check text {}: '{}'", i, text) << std::endl;
                save_image(fmt::format("pics/{}.png", i++), cut.mat);
                if (text.contains("prime")) {
                    if (items_slugs.contains(text)) {
                        auto &item_info = items_slugs[text];
                        auto orders = get_prices(item_info.slug);
                        std::cout << fmt::format("{: <40}", text);
                        std::cout << " sell|buy: ";
                        for (auto &v: orders.sell) {
                            if (v == 4096 || v == 0) {
                                std::cout << "   ";
                            } else {
                                std::cout << fmt::format("{: >3}", v);
                            }
                        }
                        std::cout << " | ";
                        for (auto &v: orders.buy) {
                            if (v == 4096 || v == 0) {
                                std::cout << "   ";
                            } else {
                                std::cout << fmt::format("{: >3}", v);
                            }
                        }
                        if (item_info.vaulted) {
                            std::cout << " VAULTED";
                        }
                        std::cout << std::endl;
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    } else {
                        std::cout << fmt::format("cant find item in cache '{}'", text) << std::endl;
                    }
                }
            }
        }
    });
    return 0;
}
