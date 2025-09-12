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

static bool record_frame = false;
const std::string base_path = "/home/gudov/src/warfeye";

struct ProcessedImgs {
    cv::Mat counters;
    cv::Mat masked;
};

ProcessedImgs processFrame(cv::Mat &&img) {
    ProcessedImgs imgs;
    cv::Mat filtered_small;
    cv::threshold(img, filtered_small, 128, 256, 0);
    cv::inRange(filtered_small, cv::Scalar(224,224,224), cv::Scalar(256,256,256), filtered_small);

    cv::Mat filtered_big;
    cv::inRange(img, cv::Scalar(224,224,224), cv::Scalar(256,256,256), filtered_big);
    cv::blur(filtered_big, filtered_big, cv::Size(3,3));
    cv::threshold(filtered_big, filtered_big, 48, 256, 0);
    cv::blur(filtered_big, filtered_big, cv::Size(5,5));
    cv::threshold(filtered_big, filtered_big, 32, 256, 0);

    cv::copyTo(filtered_small, imgs.masked, filtered_big);
    cv::threshold(imgs.masked, imgs.masked, 128, 256, 1);

    // cv::blur(filtered_big, counter, cv::Size(3,3));
    // cv::threshold(counter, counter, 1, 256, 0);
    cv::threshold(filtered_big, imgs.counters, 1, 256, 0);

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

std::optional<std::string> recognizeCut(Cut &cut, tesseract::TessBaseAPI *api, const std::unordered_set<std::string> &filter, const std::unordered_set<std::string> &exclude) {
    api->SetImage((uchar*)cut.mat.data, cut.mat.size().width, cut.mat.size().height, cut.mat.channels(), cut.mat.step1());
    api->Recognize(0);

    std::string text;
    bool filter_pass = false;
    tesseract::ResultIterator* ri = api->GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
    if (ri != 0) {
        do {
            const char* word = ri->GetUTF8Text(level);
            if (word) {
                std::string word_str = word;
                std::transform(word_str.begin(), word_str.end(), word_str.begin(),
                    [](unsigned char c){ return std::tolower(c); });
                if (filter.contains(word_str)) {
                    filter_pass = true;
                }
                if (!exclude.contains(word_str)) {
                    if (!text.empty()) {
                        text += " ";
                    }
                    text += word_str;
                }
            }
            delete[] word;
        } while (ri->Next(level));
    }

    return filter_pass ? std::make_optional(text) : std::nullopt;
}

int main(int argc, char *argv[]) {
    std::thread keyboard([](){
        Display*    dpy     = XOpenDisplay(0);
        Window      root    = DefaultRootWindow(dpy);
        XEvent      ev;

        unsigned int    modifiers       = ControlMask | ShiftMask;
        int             keycode         = XKeysymToKeycode(dpy,XK_P);
        Window          grab_window     =  root;
        Bool            owner_events    = False;
        int             pointer_mode    = GrabModeAsync;
        int             keyboard_mode   = GrabModeAsync;

        XGrabKey(dpy, keycode, modifiers, grab_window, owner_events, pointer_mode,
                keyboard_mode);

        XSelectInput(dpy, root, KeyPressMask );
        while(true) {
            XNextEvent(dpy, &ev);
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
    FILE *all_items_f = fopen("all_items.json", "wb");
    fwrite(all_items.text.data(), all_items.text.size(), 1, all_items_f);
    fclose(all_items_f);
    exit(0);

    init_screencast(argc, argv, [&](void*data,uint32_t size,size_t w,size_t h) {
        if (record_frame) {
            record_frame = false;
            std::cout << "image recieved" << std::endl;
            cv::Mat image(h,w, CV_8UC4, (uint8_t*)data);
            cv::Mat image_rgb;
            cv::cvtColor(image, image_rgb, cv::COLOR_RGBA2RGB);
            auto processed = processFrame(std::move(image_rgb));
            auto cuts = cutImages(processed);

            for (auto &cut: cuts) {
                auto text = recognizeCut(cut, tess_api, filter, exclude);
                if (text) {
                    std::cout << *text << std::endl;
                }
            }
        }
    });
    return 0;
}
