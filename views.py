#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:02:46 2023

@author: oriontomasi
"""

import asyncio
import aiofiles
from deep_translator import GoogleTranslator
import ast
import os
import io
import config as settings
from PIL import Image
import math
import json

import aiohttp
import asyncio

async def url2pil(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            image_data = await response.read()
            image = Image.open(io.BytesIO(image_data))
            return image
        
async def ar2size(aspect_ratio, smaller_size):
    width, height = aspect_ratio.split(':')
    width, height = int(width), int(height)
    
    if width < height:
        new_width = smaller_size
        new_height = min(math.ceil((height / width) * smaller_size), 3072)
    else:
        new_height = smaller_size
        new_width = min(math.ceil((width / height) * smaller_size), 3072)
        
    new_width = round(new_width/64)*64
    new_height = round(new_height/64)*64
    return new_width, new_height

async def add_data(data):
    async with aiofiles.open('persistence.txt', mode='a') as f:
        await f.write(data+'\n')
        
async def trans_query(text):
    textr = GoogleTranslator(source='auto', target='en').translate(text)
    print(textr)
    return textr

async def upscale_code(code, number):
    filepath = 'upscalecache/' + code + '_' + str(number) + settings.img_type
    
    with open('textcache/' + code + '.txt', 'r') as f:
        text = f.read()
        amount = ast.literal_eval(text)['amount']
    
    if os.path.isfile(filepath):
        return filepath
    
    #img = Image.open('imagecache/' + code + settings.img_type)
    with open('textcache/' + code + '-url.txt', 'r') as f:
        imageurl = f.read()
        
    img = await url2pil(imageurl)
    
    grid_width = 2
    grid_height = 2
    
    if amount == 2:
        grid_height = 1
    elif amount == 6:
        grid_width = 3
    elif amount == 8:
        grid_width = 4
    elif amount == 9:
        grid_width = 3
        grid_height = 3
    
    cell_width = img.width // grid_width
    cell_height = img.height // grid_height
    
    row = (number-1) // grid_width
    col = (number-1) % grid_width
    left = col * cell_width
    top = row * cell_height
    right = left + cell_width
    bottom = top + cell_height
        
    print(left, top, right, bottom)
    
    img = img.crop((left, top, right, bottom))
    img.save(filepath)
    return filepath

async def make_grid(images, amount, width, height):
    im0 = images[0]    
    if amount == 1:
        return im0
    
    elif amount == 2:
        dst = Image.new('RGB', (width*2, height))
        
        dst.paste(images[0], (0,0))
        dst.paste(images[1], (width, 0))
        
    elif amount == 4:
        dst = Image.new('RGB', (width*2, height*2))
        
        dst.paste(images[0], (0,0))
        dst.paste(images[1], (width, 0))
        
        dst.paste(images[2], (width, height))
        dst.paste(images[3], (0, height))
        
    elif amount == 6:
        dst = Image.new('RGB', (width*3, height*2))
        
        dst.paste(images[0], (0,0))
        dst.paste(images[1], (0, height))
        
        dst.paste(images[2], (width, 0))
        dst.paste(images[3], (width, height))
        
        dst.paste(images[4], (width*2, 0))
        dst.paste(images[5], (width*2, height))
        
    elif amount == 8:
        dst = Image.new('RGB', (width*4, height*2))
        
        dst.paste(images[0], (0,0))
        dst.paste(images[1], (width, 0))
        dst.paste(images[2], (width*2,0))
        dst.paste(images[3], (width*3, 0))
        
        dst.paste(images[4], (0,height))
        dst.paste(images[5], (width, height))
        dst.paste(images[6], (width*2,height))
        dst.paste(images[7], (width*3, height))
        
    else:
        dst = Image.new('RGB', (width*3, height*3))
        
        dst.paste(images[0], (0,0))
        dst.paste(images[1], (width, 0))
        dst.paste(images[2], (width*2,0))
        
        dst.paste(images[3], (0,height))
        dst.paste(images[4], (width, height))
        dst.paste(images[5], (width*2,height))
        
        dst.paste(images[6], (0,height*2))
        dst.paste(images[7], (width, height*2))
        dst.paste(images[8], (width*2,height*2))
    return dst

async def average_color(path):
    try:
        im = Image.open(path)
    except:
        im = path
    
    width, height = im.size
    
    im = im.resize((width//16, height//16))
    
    width, height = im.size
    
    pixels = list(im.getdata())
    n = len(pixels)
    r_sum = g_sum = b_sum = 0
    for pixel in pixels:
        r, g, b = pixel
        r_sum += r
        g_sum += g
        b_sum += b
        
    average_color = (r_sum//n, g_sum//n, b_sum//n)
    
    high = max(average_color)
    low = min(average_color)
    average_color = tuple([min(255, max(0, x + (-100 if x == low else 100 if x == high else 0))) for x in average_color])
    
    return (average_color[0] << 16) + (average_color[1] << 8) + average_color[2]