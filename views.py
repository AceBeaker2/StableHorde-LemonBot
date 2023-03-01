#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:02:46 2023

@author: oriontomasi
"""

import asyncio
import aiofiles
from deep_translator import GoogleTranslator

async def add_data(data):
    async with aiofiles.open('persistence.txt', mode='a') as f:
        await f.write(data+'\n')
        
async def trans_query(text):
    textr = GoogleTranslator(source='auto', target='en').translate(text)
    print(textr)
    return textr

