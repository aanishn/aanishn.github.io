---
layout: post
title: TOTP Viewer Python PyOTP PySimpleGUI
date: '2023-05-02'
author: "aanishn"
tags:
- TOTP, Python, PySimpleGUI, pyotp
---


```python
#!/usr/bin/env python

#pip install pysimplegui

import datetime

import pyotp
import PySimpleGUI as sg

totp_list = {
    "Sample 1": "4XVEKNAPFBIGNCX5NGA2DE3GTAM3UPKV",
    "Sample 2": "DWX74LBWAAPIJXHCZQFATLWLIYUUBCXQ",
    "Sample 3": "45F7FUDKSH6WT4FZBC26HLXE6MJLLI5E",
}

totp_objs = { k:(pyotp.TOTP(v), sg.Text(key=k)) for k, v in totp_list.items() }

sg.theme('BluePurple') 

layout = [ [sg.Text(f"{k} => "), v[1], sg.ProgressBar(30, orientation='h', size=(10, 15), key=f"{k}_progress")] for k, v in totp_objs.items() ]
layout.append([sg.Button('Quit')])

# Create the Window
window = sg.Window('TOTP Viewer', layout)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read(timeout=1)
    if event == sg.WIN_CLOSED or event == 'Quit': # if user closes window or clicks cancel
        break
    else:
        for k, v in totp_objs.items():
            window[k].update(f"{v[0].now()}")
            pr = (datetime.datetime.now().second % v[0].interval)
            window[f"{k}_progress"].update_bar(pr)

window.close()
```