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

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "json.hpp"
#include "utf8.h"

struct Config {
    bool display_orders = true;
    bool display_ducats = false;
    bool ru = false;
    float text_box_bonus_mult = 1;
};

Config config;
void ReloadConfig() {
    // if (!std::filesystem::exists("config.json")) {
    //     nlohmann::json def;
    //     def["display_orders"] = true;
    //     def["display_ducats"] = false;
    //     def["text_box_bonus_mult"] = 1.f;
    //     std::ofstream file("config.txt");
    //     file << def.dump();
    //     file.flush();
    // }

    std::ifstream file("config.json");
    std::ostringstream sstr;
    sstr << file.rdbuf();
    nlohmann::json config_ = nlohmann::json::parse(sstr.str());
    config.display_orders = config_["display_orders"];
    config.display_ducats = config_["display_ducats"];
    config.text_box_bonus_mult = config_["text_box_bonus_mult"];
    config.ru = config_["ru"];
}

constexpr bool save_frames_to_disk = true;
template<typename mat>
void save_image(std::string path, mat m) {
    if constexpr (save_frames_to_disk) {
        cv::imwrite(path, m);
    }
}

static bool record_frame = true;
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
        boundRect[i].width = std::min(boundRect[i].width + int(boundRect[i].height * config.text_box_bonus_mult), max_w);
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
        it->width = std::min(it->width, imgs.masked.cols - it->x - 1);
        it->height = std::min(it->height, imgs.masked.rows - it->y - 1);
        cuts.push_back({
            .pos = *it,
            .mat = imgs.masked(*it).clone()
        });
        save_image(fmt::format("pics/{}.png", i), cuts.back().mat);
        it++;
    }

    return cuts;
}

tesseract::TessBaseAPI *initTesseract() {
    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    const std::string lang = config.ru ? "rus" : "eng";
    if (api->Init("/usr/share/tessdata/", lang.c_str(), tesseract::OEM_LSTM_ONLY)) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    api->SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    static std::string allowed;
    auto add_range = [&](char begin, char end) {
        while (begin <= end) {
            allowed += begin;
            begin++;
        }
    };
    if (config.ru) {
        allowed += "ёйцукенгшщзхъфывапролджэячсмитьбю";
        allowed += "ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ";
    } else {
        add_range('A', 'Z');
        add_range('a', 'z');
    }
    add_range('0', '9');
    allowed += ' ';
    allowed += '&';
    api->SetVariable("tessedit_char_whitelist", allowed.c_str());

    return api;
}

std::unordered_map<uint32_t, uint32_t> lower_cache;
uint32_t tolower_m(uint32_t ch) {
    if (lower_cache.empty()) {
        std::u8string lower = u8"еейцукенгшщзхъфывапролджэячсмитьбюqwertyuiopasdfghjklzxcvbnm";
        std::u8string upper = u8"ЁёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮQWERTYUIOPASDFGHJKLZXCVBNM";
        std::u32string lower_u32;
        std::u32string upper_u32;
        utf8::utf8to32(lower.begin(), lower.end(), std::back_inserter(lower_u32));
        utf8::utf8to32(upper.begin(), upper.end(), std::back_inserter(upper_u32));
        for (size_t i = 0; i < lower.size(); i++) {
            lower_cache[upper_u32[i]] = lower_u32[i];
        }
    }
    auto it = lower_cache.find(ch);
    return (it != lower_cache.end()) ? it->second : ch;
}

std::set<uint32_t> chars_blacklist{':', '(', ')'};
std::string tolower_utf(std::string str) {
    std::u8string word_u8{str.begin(), str.end()};
    std::u32string word_u32;
    utf8::utf8to32(word_u8.begin(), word_u8.end(), std::back_inserter(word_u32));
    std::u32string lower_u32;
    for (auto &ch : word_u32) {
        if (!chars_blacklist.contains(ch)) {
            lower_u32.push_back(tolower_m(ch));
        }
    }
    word_u8.clear();
    utf8::utf32to8(lower_u32.begin(), lower_u32.end(), std::back_inserter(word_u8));
    return std::string{word_u8.begin(), word_u8.end()};
}

std::u8string recognizeCut(Cut &cut, tesseract::TessBaseAPI *api) {
    api->SetImage((uchar*)cut.mat.data, cut.mat.size().width, cut.mat.size().height, cut.mat.channels(), cut.mat.step1());
    api->Recognize(0);

    std::u8string text;
    tesseract::ResultIterator* ri = api->GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
    if (ri != 0) {
        do {
            const char* word = ri->GetUTF8Text(level);
            if (word) {
                std::string word_str = word;
                std::u8string word_u8{word_str.begin(), word_str.end()};
                std::u32string word_u32;
                utf8::utf8to32(word_u8.begin(), word_u8.end(), std::back_inserter(word_u32));
                for (auto &ch : word_u32) {
                    ch = tolower_m(ch);
                }
                word_u8.clear();
                utf8::utf32to8(word_u32.begin(), word_u32.end(), std::back_inserter(word_u8));
                if (text != u8"jy") {
                    if (!text.empty()) {
                        text += u8" ";
                    }
                    text += word_u8;
                }
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

    std::sort(orders.sell.begin(), orders.sell.end(), [](int a, int b){return a < b;});
    std::sort(orders.buy.begin(), orders.buy.end(), [](int a, int b){return a > b;});
    return orders;
}

struct ItemInfo {
    std::string slug;
    bool vaulted;
    std::optional<int> ducats;
};

int main(int argc, char *argv[]) {
    ReloadConfig();
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
        "relic",
        "прайм"
    };

    std::unordered_set<std::string> exclude = {
        "[radiant]",
        "[flawless]"
    };

    cpr::Response all_items = cpr::Get(cpr::Url("https://api.warframe.market/v2/items"), cpr::Header{{"Language", "ru"}});

    auto items_json = nlohmann::json::parse(all_items.text);
    auto items_data = items_json["data"];
    std::unordered_map<std::string, ItemInfo> items_slugs;
    for (auto item: items_data) {
        std::string name = item["i18n"][config.ru ? "ru" : "en"]["name"];
        name = tolower_utf(name);
        // std::cout << name << std::endl;
        items_slugs[name] = {
            .slug = item["slug"],
            .vaulted = item["vaulted"].is_boolean() ? (bool)item["vaulted"] : false,
            .ducats = item["ducats"].is_number() ? std::make_optional(int(item["ducats"])) : std::nullopt
        };
        if (name.ends_with(" чертеж")) {
            std::string old_name = name;
            size_t pos = name.find(" чертеж");
            name = std::string("чертеж ") + name.substr(0, pos);
            // std::cout << name << std::endl;
            items_slugs[name] = items_slugs[old_name];
        }
    }

    init_screencast(argc, argv, [&](void*data,uint32_t size,size_t w,size_t h) {
        if (record_frame) {
            ReloadConfig();
            record_frame = false;
            std::cout << "image recieved" << std::endl;
            cv::Mat image(h,w, CV_8UC4, (uint8_t*)data);
            cv::Mat image_rgb;
            cv::cvtColor(image, image_rgb, cv::COLOR_RGBA2RGB);
            auto processed = processFrame(std::move(image_rgb));
            auto cuts = cutImages(processed);

            int total_plt = 0;
            int total_ducats = 0;
            std::unordered_map<int, int> ducats_plt = {
                {15, 1},
                {25, 1},
                {45, 2},
                {65, 3},
                {100, 7}
            };

            for (auto &cut: cuts) {
                auto text_u8 = recognizeCut(cut, tess_api);
                std::string text{text_u8.begin(), text_u8.end()};
                std::cout << text << std::endl;
                if (text.contains("prime") || text.contains("прайм")) {
                    if (items_slugs.contains(text)) {
                        auto &item_info = items_slugs[text];
                        std::cout << fmt::format("{: <40}", text);

                        if (config.display_orders) {
                            auto orders = get_prices(item_info.slug);
                            std::this_thread::sleep_for(std::chrono::seconds(1));
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
                        }
                        if (config.display_ducats && item_info.ducats) {
                            int item_ducats = *item_info.ducats;
                            std::cout << fmt::format("{: >4}", item_ducats);
                            int item_ducat_plt = ducats_plt[item_ducats];
                            std::cout << fmt::format("{: >3}", item_ducat_plt);
                            total_ducats += item_ducats;
                            total_plt += item_ducat_plt;
                        }
                        if (item_info.vaulted) {
                            std::cout << " VAULTED";
                        }
                        std::cout << std::endl;
                    } else {
                        std::cout << fmt::format("cant find item in cache '{}'", text) << std::endl;
                    }
                }
            }
            if (total_plt != 0) {
                std::cout << fmt::format("plt for ducats: {}, ducats: {}", total_plt, total_ducats) << std::endl;
            }
        }
    });
    return 0;
}
