---
layout: post
title: TOTP Viewer Python PyOTP PySimpleGUI
date: '2025-04-19'
author: "aanishn"
tags:
- TOTP, Python, PySimpleGUI, pyotp
---

TOTP Viewer v2

Improvements,
1. Theme and UX
2. Supports add/edit/delete keys
3. Copy TOTP to clipboard

```python
#!~/miniconda3/envs/pysimplegui/bin/python

#pip install pysimplegui
#pip install pyotp
#pip install pyperclip

import pyotp
import PySimpleGUI as sg
import yaml
import threading
import time
import datetime
import pyperclip
from pathlib import Path

class TOTPManager:
    def __init__(self):
        self.config_file = Path('totp_config.yaml')
        self.totp_list = self.load_config()
        self.totp_objs = {}
        self.setup_totp_objects()
        
    def load_config(self):
        if not self.config_file.exists():
            return {}
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f) or {}
            
    def save_config(self):
        with open(self.config_file, 'w') as f:
            yaml.dump(self.totp_list, f)
            
    def setup_totp_objects(self):
        self.totp_objs = {
            k: (pyotp.TOTP(v), sg.Text(size=(6, 1), key=k, font=('Helvetica', 12, 'bold'), enable_events=True)) 
            for k, v in self.totp_list.items()
        }
        
    def add_totp(self, name, secret):
        self.totp_list[name] = secret
        self.save_config()
        self.setup_totp_objects()
        
    def delete_totp(self, name):
        if name in self.totp_list:
            del self.totp_list[name]
            self.save_config()
            self.setup_totp_objects()
            
    def edit_totp(self, old_name, new_name, new_secret):
        if old_name in self.totp_list:
            del self.totp_list[old_name]
            self.totp_list[new_name] = new_secret
            self.save_config()
            self.setup_totp_objects()

class TOTPViewer:
    def __init__(self):
        self.manager = TOTPManager()
        self.window = None
        sg.theme('DarkBlue')
        self.stop_thread = False
        self.totp_thread = None
        
    def create_layout(self):
        totp_rows = []
        for k, v in self.manager.totp_objs.items():
            row = [
                sg.Text(f"{k}", size=(30, 1), font=('Helvetica', 11)),
                sg.Text("=>", font=('Helvetica', 11)),
                v[1],
                sg.ProgressBar(30, orientation='h', size=(20, 15), key=f"{k}_progress"),
                sg.Button('Edit', key=f"EDIT_{k}"),
                sg.Button('Delete', key=f"DELETE_{k}")
            ]
            totp_rows.append(row)
            
        layout = [
            [sg.Text('TOTP Viewer', font=('Helvetica', 16, 'bold'))],
            [sg.HorizontalSeparator()],
            *totp_rows,
            [sg.HorizontalSeparator()],
            [sg.Button('Add New TOTP'), sg.Button('Quit')]
        ]
        return layout
        
    def update_totps(self):
        while not self.stop_thread:
            for k, v in self.manager.totp_objs.items():
                try:
                    totp_value = v[0].now()
                    progress = datetime.datetime.now().second % v[0].interval
                    self.window.write_event_value('-TOTP_UPDATE-', (k, totp_value, progress))
                except Exception as e:
                    print(f"Error updating TOTP {k}: {e}")
            time.sleep(1)
            
    def show_add_edit_window(self, edit_mode=False, name="", secret=""):
        title = "Edit TOTP" if edit_mode else "Add New TOTP"
        layout = [
            [sg.Text('Name:', size=(12, 1)), sg.Input(name, key='name')],
            [sg.Text('Secret:', size=(12, 1)), sg.Input(secret, key='secret')],
            [sg.Button('Save'), sg.Button('Cancel')]
        ]
        window = sg.Window(title, layout, modal=True)
        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, 'Cancel'):
                break
            if event == 'Save':
                if values['name'] and values['secret']:
                    if edit_mode:
                        self.manager.edit_totp(name, values['name'], values['secret'])
                    else:
                        self.manager.add_totp(values['name'], values['secret'])
                    break
                else:
                    sg.popup('Please fill in all fields', title='Error')
        window.close()
        
    def run(self):
        self.window = sg.Window('TOTP Viewer', self.create_layout(), finalize=True)
        
        # Start TOTP update thread
        self.totp_thread = threading.Thread(target=self.update_totps, daemon=True)
        self.totp_thread.start()
        
        while True:
            event, values = self.window.read()
            
            if event == sg.WIN_CLOSED or event == 'Quit':
                self.stop_thread = True
                break
                
            elif event == 'Add New TOTP':
                self.show_add_edit_window()
                self.window.close()
                self.window = sg.Window('TOTP Viewer', self.create_layout(), finalize=True)
                
            elif event.startswith('DELETE_'):
                name = event.replace('DELETE_', '')
                if sg.popup_yes_no(f'Delete TOTP for {name}?', title='Confirm Delete') == 'Yes':
                    self.manager.delete_totp(name)
                    self.window.close()
                    self.window = sg.Window('TOTP Viewer', self.create_layout(), finalize=True)
                    
            elif event.startswith('EDIT_'):
                name = event.replace('EDIT_', '')
                self.show_add_edit_window(True, name, self.manager.totp_list[name])
                self.window.close()
                self.window = sg.Window('TOTP Viewer', self.create_layout(), finalize=True)
                
            elif event in self.manager.totp_objs:
                # Copy TOTP to clipboard when clicked
                totp_value = self.manager.totp_objs[event][0].now()
                pyperclip.copy(totp_value)
                sg.popup_quick_message('Copied to clipboard!', auto_close_duration=1)
                
            elif event == '-TOTP_UPDATE-':
                k, totp_value, progress = values[event]
                if k in self.manager.totp_objs:
                    self.window[k].update(totp_value)
                    self.window[f"{k}_progress"].update(progress)
        
        self.window.close()

if __name__ == '__main__':
    viewer = TOTPViewer()
    viewer.run()
```
