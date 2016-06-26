---
layout: post
title: Android - Dump SMS
date: '2016-06-26'
author: "aanishn"
tags:
- android
---

### [QPython3](https://play.google.com/store/apps/details?id=com.hipipal.qpy3&hl=en) Android script to dump SMS 

```
import pickle
import androidhelper as android

droid = android.Android()

mids = droid.smsGetMessageIds(False, "inbox")

message = []
for id in mids[1]:
	m = droid.smsGetMessageById(id)
	message.append((m[0], m[1]))

try:
	fp = open('/mnt/sdcard/sms.pkl', 'wb')
	pickle.dump(message, fp)
	print("file saved...")
except:
	print("error opening file...")
```
