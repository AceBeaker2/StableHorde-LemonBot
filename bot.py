import disnake
from typing import Optional
from disnake.ext import commands
import config as settings
import nest_asyncio
# above line is weird spyder workaround. DW about it.
import asyncio
import aiohttp
import time
import logging
from PIL import Image, ImageDraw
from nsfw_detector import predict
import base64
import io
import ast
import aiofiles
import re

import argparse

from random import randint

import os
import json

import views

try:
    import intents
except:
    print('ERROR: CP checker not imported')

model = predict.load_model('./checker.h5')
# gets model

logging.basicConfig(level=logging.INFO)
nest_asyncio.apply()
# weird spyder thing I don't get it

class UpscaleView(disnake.ui.View):
    def __init__(self, codeid, number):
        super().__init__(timeout=None)
        with open('textcache/' + codeid + '.txt', 'r') as f:
            text = f.read()
            
        self.codeid = codeid
        self.vars = ast.literal_eval(text)
        self.number = number
        filepath = asyncio.run(views.upscale_code(codeid, number))
        self.filepath = filepath
        self.color = asyncio.run(views.average_color(self.filepath))
        
        self.moreinfo.custom_id = f'{codeid}@{number}@m'
        
        self.facial.custom_id = f'{codeid}@{number}@f'
        
        self.variate.custom_id = f'{codeid}@{number}@v'
            
    @disnake.ui.button(custom_id='info', label='More Info!', style=disnake.ButtonStyle.blurple, emoji=settings.thinking_emoji, row=1)
    async def moreinfo(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        embed=disnake.Embed(title='Status Sheet', color=self.color)
        
        embed.add_field(name='Seed:', value=self.vars.get(str(self.number-1), -1).get('seed', -1), inline=False)
        embed.add_field(name='Seed:', value=self.vars['0']['seed'], inline=False)
        embed.add_field(name='Sampler:', value=self.vars['sampler'], inline=False)
        embed.add_field(name='Prompt:', value=self.vars['prompt'][:1024], inline=False)
        embed.add_field(name='Negative Prompt:', value=self.vars['neg_prompt'][:1024], inline=False)
        embed.add_field(name='Model:', value=self.vars['model'], inline=False)
        embed.add_field(name='Content Filter', value=str(self.vars['filter']), inline=False)
        embed.add_field(name='CFG Scale:', value=self.vars['cfg_scale'], inline=False)
        embed.add_field(name='Steps:', value=str(self.vars['steps']), inline=False)
        embed.add_field(name='Dimensions:', value=str(self.vars['width']) + 'x' + str(self.vars['height']), inline=False)
        embed.add_field(name='Upscaler:', value=str(self.vars['upscalers']), inline=False)
        embed.add_field(name='Author:', value='<@'+str(self.vars['author'])+'>', inline=False)
            
        embed.set_image(file=disnake.File(self.filepath))
        await inter.send(embed=embed)

    @disnake.ui.button(custom_id='esrgan', label='x4 Upscale', style=disnake.ButtonStyle.blurple, emoji=settings.eyes_emoji, row=1)
    async def esrgan(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        await inter.send('Your Upscale is on the way!', ephemeral=True) 
        dm = False
        try:
            nsfw = not inter.channel.nsfw or settings.nsfw_filter
        except AttributeError:
            dm = True
            nsfw = True
                   
        image_info = await pil2base64(Image.open(self.filepath))
        source_image, width, height = image_info
        author = inter.user

        inter_response = await inter.original_response()         
            
        vvars = {
                 'prompt': self.vars['prompt'],
                 'raw_prompt': ' **x4 Upscale** ',
                 'neg_prompt': self.vars['neg_prompt'],
                 'width': width,
                 'height': height,
                 'upscalers': 'RealESRGAN_x4plus',
                 'model': 'stable_diffusion',
                 'cfg_scale': self.vars['cfg_scale'],
                 'userid': author.mention,
                 'channelid': inter.channel_id,
                 'sampler': self.vars['sampler'],
                 'karras': True,
                 'steps': 70,
                 'source_image': source_image,
                 'source_processing': 'img2img',
                 'source_mask': None,
                 'filter':nsfw,
                 'dm': dm,
                 'author': author,
                 'denoising_strength': 0.05,
                 'seed': -1,
                 'inter_response': inter_response,
                 'amount' : 1,
                 'tiling' : False,
                 'hires_fix' : False,
                 'english' : True,
                 'midj' : False,
                }
        await query_api(settings.url + settings.endpoint, vvars, inter)
    
        
    @disnake.ui.button(custom_id='facial', label='Facial Redo', style=disnake.ButtonStyle.blurple, emoji=settings.pleading_emoji, row=1)
    async def facial(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        await inter.send('Your Facial Redo is on the way!', ephemeral=True) 
        dm = False
        try:
            nsfw = not inter.channel.nsfw or settings.nsfw_filter
        except AttributeError:
            dm = True
            nsfw = True
                   
        image_info = await pil2base64(Image.open(self.filepath))
        source_image, width, height = image_info
        author = inter.user

        inter_response = await inter.original_response()         
            
        vvars = {
                 'prompt': self.vars['prompt'],
                 'raw_prompt': ' **Facial Redo** ',
                 'neg_prompt': self.vars['neg_prompt'],
                 'width': self.vars['width'],
                 'height': self.vars['height'],
                 'upscalers': 'CodeFormers',
                 'model': self.vars['model'],
                 'cfg_scale': self.vars['cfg_scale'],
                 'userid': author.mention,
                 'channelid': inter.channel_id,
                 'sampler': self.vars['sampler'],
                 'karras': True,
                 'steps': 10,
                 'source_image': source_image,
                 'source_processing': 'img2img',
                 'source_mask': None,
                 'filter':nsfw,
                 'dm': dm,
                 'author': author,
                 'denoising_strength': 0.05,
                 'seed': -1,
                 'inter_response': inter_response,
                 'amount' : 1,
                 'tiling' : False,
                 'hires_fix' : False,
                 'english' : False,
                 'midj' : False,
                }
        await query_api(settings.url + settings.endpoint, vvars, inter)
    

    @disnake.ui.button(custom_id='variate', label='Make Variations', style=disnake.ButtonStyle.green)
    async def variate(self, button: disnake.ui.Button, inter: disnake.MessageInteraction):
        await inter.send('Your Variation is on the way!', ephemeral=True) 
        dm = False
        try:
            nsfw = not inter.channel.nsfw or settings.nsfw_filter
        except AttributeError:
            dm = True
            nsfw = True
                   
        image_info = await pil2base64(Image.open(self.filepath))
        source_image, width, height = image_info
        
        author = inter.user

        inter_response = await inter.original_response()         
        
        vvars = {
                 'prompt': self.vars['prompt'],
                 'raw_prompt': self.vars['raw_prompt'],
                 'neg_prompt': self.vars['neg_prompt'],
                 'width': self.vars['width'],
                 'height': self.vars['height'],
                 'upscalers': 'GFPGAN',
                 'model': self.vars['model'],
                 'cfg_scale': self.vars['cfg_scale'],
                 'userid': author.mention,
                 'channelid': inter.channel_id,
                 'sampler': self.vars['sampler'],
                 'karras': True,
                 'steps': 10,
                 'source_image': source_image,
                 'source_processing': 'img2img',
                 'source_mask': None,
                 'filter':nsfw,
                 'dm': dm,
                 'author': author,
                 'denoising_strength': 0.4,
                 'seed': -1,
                 'inter_response': inter_response,
                 'amount': settings.default_images,
                 'tiling' : False,
                 'hires_fix' : True,
                 'english' : False,
                 'midj' : self.vars.get('midj', False),
                 }
        await query_api(settings.url + settings.endpoint, vvars, inter)
    
class GenView(disnake.ui.View):
    def make_view(self, amount, codeid, midj=False):
    
        u1 = disnake.ui.Button(custom_id=codeid+'_u1', label='U1', style=disnake.ButtonStyle.blurple, row=0)
        u2 = disnake.ui.Button(custom_id=codeid+'_u2', label='U2', style=disnake.ButtonStyle.blurple, row=0)
        u3 = disnake.ui.Button(custom_id=codeid+'_u3', label='U3', style=disnake.ButtonStyle.blurple, row=0)
        u4 = disnake.ui.Button(custom_id=codeid+'_u4', label='U4', style=disnake.ButtonStyle.blurple, row=0)
        u5 = disnake.ui.Button(custom_id=codeid+'_u5', label='U5', style=disnake.ButtonStyle.blurple, row=1)
        u6 = disnake.ui.Button(custom_id=codeid+'_u6', label='U6', style=disnake.ButtonStyle.blurple, row=1)
        u7 = disnake.ui.Button(custom_id=codeid+'_u7', label='U7', style=disnake.ButtonStyle.blurple, row=1)
        u8 = disnake.ui.Button(custom_id=codeid+'_u8', label='U8', style=disnake.ButtonStyle.blurple, row=1)
        u9 = disnake.ui.Button(custom_id=codeid+'_u9', label='U9', style=disnake.ButtonStyle.blurple, row=1)
    
        if amount == 1:
            u1 = disnake.ui.Button(custom_id=codeid+'_u1', label='U1', style=disnake.ButtonStyle.blurple, row=0, disabled=True)
        elif amount == 4:
            u3 = disnake.ui.Button(custom_id=codeid+'_u3', label='U3', style=disnake.ButtonStyle.blurple, row=1)
            u4 = disnake.ui.Button(custom_id=codeid+'_u4', label='U4', style=disnake.ButtonStyle.blurple, row=1)
        elif amount == 6:
            u4 = disnake.ui.Button(custom_id=codeid+'_u4', label='U4', style=disnake.ButtonStyle.blurple, row=1)
        elif amount == 9:
            u4 = disnake.ui.Button(custom_id=codeid+'_u4', label='U4', style=disnake.ButtonStyle.blurple, row=1)
            u7 = disnake.ui.Button(custom_id=codeid+'_u7', label='U7', style=disnake.ButtonStyle.blurple, row=2)
            u8 = disnake.ui.Button(custom_id=codeid+'_u8', label='U8', style=disnake.ButtonStyle.blurple, row=2)
            u9 = disnake.ui.Button(custom_id=codeid+'_u9', label='U9', style=disnake.ButtonStyle.blurple, row=2)
        
        regen = disnake.ui.Button(custom_id=codeid+'_regen', emoji=settings.redo_emoji, style=disnake.ButtonStyle.blurple, row=1)
        
        async def regencallback(interaction):
            with open('textcache/' + codeid + '.txt', 'r') as f:
                text = f.read()
            rvars = ast.literal_eval(text)
            await imagine(interaction, rvars['raw_prompt'])
        async def u1callback(interaction):
            await upscale(interaction, number=1, code=codeid)
        async def u2callback(interaction):
            await upscale(interaction, number=2, code=codeid)
        async def u3callback(interaction):
            await upscale(interaction, number=3, code=codeid) 
        async def u4callback(interaction):
            await upscale(interaction, number=4, code=codeid)
        async def u5callback(interaction):
            await upscale(interaction, number=5, code=codeid)
        async def u6callback(interaction):
            await upscale(interaction, number=6, code=codeid)
        async def u7callback(interaction):
            await upscale(interaction, number=7, code=codeid)
        async def u8callback(interaction):
            await upscale(interaction, number=8, code=codeid)  
        async def u9callback(interaction):
            await upscale(interaction, number=9, code=codeid)
    
        regen.callback = regencallback
        u1.callback = u1callback
        u2.callback = u2callback
        u3.callback = u3callback
        u4.callback = u4callback
        u5.callback = u5callback
        u6.callback = u6callback
        u7.callback = u7callback
        u8.callback = u8callback
        u9.callback = u9callback
    
        view = disnake.ui.View(timeout=None)
        if amount >= 1:
            view.add_item(u1)
        if amount >= 2:
            view.add_item(u2)    
        if amount >= 4:
            view.add_item(u3)
            view.add_item(u4)   
        if amount >= 6:
            view.add_item(u5)
            view.add_item(u6)  
        if amount >= 8:
            view.add_item(u7)
            view.add_item(u8)
        if amount >= 9:
            view.add_item(u9)
        if midj:
            view.add_item(regen)
        return view
   
    
class PersistentViewBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix=commands.when_mentioned)

    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})\n------")

bot = PersistentViewBot()

@bot.slash_command(description='Help menu!!!')
async def see_help(inter):
    embed=disnake.Embed(title='Help menu!', color=0x00ff40)
    embed.set_image(url='https://i.imgur.com/QLT48xE.png')
    
    await inter.response.send_message(embed=embed)

@bot.slash_command(description='Caption image with CLIP interrogator')
async def caption(
    inter: disnake.ApplicationCommandInteraction,
    image: disnake.Attachment = commands.Param(description='Image to caption'),
    ):
    if not str(image.content_type) == image.content_type or not image.content_type.endswith(settings.input_types):
        await inter.response.send_message('Attachment is not valid image of types: WebP, PNG, JPG, JPEG', ephemeral=True)
        return
    
    await inter.response.defer(with_message = True)
    
    image_bytes = await image.read()
    base64_bytes = base64.b64encode(image_bytes)
    base64_string = base64_bytes.decode('utf-8')
    
    json_data = {
  "forms": [{"name": "caption",}],
      "source_image": base64_string}
    
    headers = {
    # Already added when you pass json= but not when you pass data=
    # 'Content-Type': 'application/json',
        'apikey': settings.sd_api_key,
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post('https://stablehorde.net/api/v2/interrogate/async', json=json_data, headers=headers) as response:
                rawjson = await response.json()
                print(str(rawjson))
                id = rawjson['id']
                
            await asyncio.sleep(5)
            done = False
                
            while not done:
                response = await session.get('https://stablehorde.net/api/v2/interrogate/status/' + id)
                jsondata = await response.json()
                print('jsondata, I am your father:' + str(jsondata))
                try:
                    done = jsondata['state'] == 'done'
                except:
                    done = False
                await asyncio.sleep(settings.wait_time)
                
        buffer = io.BytesIO(image_bytes)
        buffer.seek(0)
        color = await views.average_color(Image.open(buffer))
        embed = disnake.Embed(title='Image Captioned!', color=color)
        embed.set_thumbnail(url=image.url)
        embed.add_field(name='Caption:', value=jsondata['forms'][0]['result']['caption'], inline=False)
        await inter.send('Interrogation done for ' + inter.author.mention + '!', embed=embed)
    except:
        await inter.send('Interrogation failed', ephemeral=True)

@bot.slash_command(description='Alternate upscale endpoint if the buttons don\'t work')
async def upscale(
    inter: disnake.ApplicationCommandInteraction,
    code: str = commands.Param(description = 'Code for upscaling', min_length=36, max_length=36),
    number: int = commands.Param(le=8, ge=1,description = 'Number to upscale'),
    ):

    try:
        await inter.response.defer(with_message=True)
    except:
        pass
    
    with open('textcache/' + code + '.txt', 'r') as f:
        text = f.read()
        vars = ast.literal_eval(text)
    
    if not number == 200:
        filepath = await views.upscale_code(code, number)
    else:
        filepath = 'imagecache/' + code + settings.img_type
    
    #try:
    color = await views.average_color(filepath)
    if not number == 200:
        embed=disnake.Embed(title='Upscale Done!', color=color)
        embed.add_field(name='Seed:', value=str(vars[str(number-1)]['seed']), inline=False)
    else:
        embed=disnake.Embed(title='Generation Done!', color=color)
        embed.add_field(name='Seed:', value=str(vars['0']['seed']), inline=False)
    embed.set_image(file=disnake.File(filepath))
    
    if not number == 200:
        view = UpscaleView(code, number)
    
    if number == 200:
        message = await inter.followup.send('Image for ' + inter.author.mention + '!',file=disnake.File(filepath))
        with open('textcache/' + code + '-url.txt', 'w') as f:
            f.write(message.attachments[0].url)
    else:
        message = await inter.followup.send('Image for ' + inter.author.mention + '!',file=disnake.File(filepath), view=view)
        with open('textcache/' + code + '-url.txt', 'w') as f:
            f.write(message.attachments[0].url)
        
    await views.add_data(code + '@' + str(number))

@bot.slash_command(description='Generates images using Stable Horde!')
async def generate(
    inter: disnake.ApplicationCommandInteraction,
    prompt: str = commands.Param(description='What the AI-generated image should be of.'),
    neg_prompt: str = commands.Param(default = '2D, grid, text', description='What the AI image model should avoid. Default: \'2D, grid, text\''),
    upscalers: str = commands.Param(choices=settings.processor_list, default='GFPGAN', description='Which Post-Processing to use for the images. Default: GFPGAN'),
    model: str = commands.Param(default=settings.default_model,autocomplete=settings.autocomp_models, description='Which model to generate the image with. Default: ' + str(settings.default_model)),
    cfg_scale: Optional[float] = commands.Param(default=8, le=30, ge=-40, description='How much the image should look like your prompt. Default: 8'),
    #imageurl: str = commands.Param(name = 'init_image', default = None, description='Initial image for img2img.'),
    width: int = commands.Param(default = settings.default_width, le=settings.max_size, ge=64, description='Width of the final image. Default: ' + str(settings.default_width)),
    height: int = commands.Param(default = settings.default_height, le=settings.max_size, ge=64, description='Height of the final image. Default: ' + str(settings.default_height)),    
    sampler: str = commands.Param(default = settings.default_sampler, description = 'ADVANCED: Which stable diffusion sampler to use. Default: ' + settings.default_sampler, choices=settings.sampler_list),
    steps: int = commands.Param(default=settings.default_steps, le=50, ge=1, description='Greater: Higher Image Quality but takes longer. Default: ' + str(settings.default_steps)),
    seed: int = commands.Param(default=-1, description='Seed for the image.'),
    amount: int = commands.Param(default=settings.default_images, choices = [1,2,4,6,8,9], description='Amount of images to generate. Default: ' + str(settings.default_images)),
    tiling: bool = commands.Param(default = False, description='Whether to have the image be repeating and tileable. Default: False'),
    hires_fix: bool = commands.Param(default = True, description='Improves Image Quality at high resolutions. Default: True'),
    english: bool = commands.Param(default = True, description='Set to false if prompt language is not English'),
): 
    raw_prompt = prompt
    if not model in settings.model_list:
        await inter.response.send_message('Invalid Model: ```'+model+'```', ephemeral=True)
        return
    
    karras = True
    dm = False
    imageurl = None
    userid = inter.author.mention
    cfg_scale = round(cfg_scale * 2)/2
    width = round(width/64)*64
    height = round(height/64)*64
    try:
        nsfw = not inter.channel.nsfw or settings.nsfw_filter
            
    except AttributeError:
        dm = True
        nsfw = True
    
    embed=disnake.Embed(title='Generation queued with following parameters:', color=0xf0e000)
    embed.set_author(name='Your generation request is on its way!')
    embed.set_thumbnail(url=settings.embed_icon)
    embed.add_field(name='Prompt:', value=prompt[:512], inline=False)
    embed.add_field(name='Negative Prompt:', value=neg_prompt[:512], inline=False)
    embed.add_field(name='CFG Scale:', value=cfg_scale, inline=True)
    embed.add_field(name='Steps:', value=steps, inline=True)
    embed.add_field(name='Model:', value=model, inline=True)
    embed.add_field(name='Content Filter:', value=str(nsfw), inline=True)
    embed.add_field(name='Dimensions:', value=str(width) + 'x' + str(height), inline=True)
    
    print(nsfw)
    
    if settings.use_embeds:
        await inter.response.send_message(embed=embed) 
    else:
        user_response = 'Generating with following parameters:\nPrompt: ```' + prompt + '```Negative Prompt: ```' + neg_prompt + '```CFG Scale: ```' + str(cfg_scale) + '```Steps: ```' + str(steps) + '```for ' + userid + '. Please be patient'
        await inter.response.send_message(user_response) 
    
    inter_response = await inter.original_response() 
    
    vars = {
        'prompt': prompt,
        'neg_prompt': neg_prompt,
        'width': width,
        'height': height,
        'upscalers': upscalers,
        'model': model,
        'cfg_scale': cfg_scale,
        'imageurl': imageurl,
        'userid': userid,
        'channelid': inter.channel_id,
        'sampler': sampler,
        'karras': karras,
        'steps': steps,
        'source_image': None,
        'source_processing': 'img2img',
        'source_mask': None,
        'filter':nsfw,
        'dm': dm,
        'author': inter.author,
        'denoising_strength': 0.75,
        'seed': seed,
        'inter_response': inter_response,
        'amount': amount,
        'tiling' : tiling,
        'hires_fix' : hires_fix,
        'english' : english,
        'midj' : False,
        'raw_prompt' : raw_prompt,
        }
    global url
    global endpoint
    await query_api(settings.url + settings.endpoint, vars, inter)
    
@bot.slash_command(description='Midjourney for brokies')
async def imagine(
    inter: disnake.ApplicationCommandInteraction,
    prompt: str = commands.Param(description='The prompt to imagine (supports midj args)'),
    ): 
    model = settings.default_model
    
    raw_prompt = prompt
    # grab raw prompt
    width = settings.default_width
    height = settings.default_height
    
    if 'https://' in raw_prompt:
        image_urls = re.findall(r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b', raw_prompt)
        if len(image_urls) != 1:
            await inter.response.send_message('Error, only one image allowed', ephemeral=True)
            return
        image_url = image_urls[0]
        print(f'Image found: {image_url}')
        prompt = prompt.split(image_url)[1]
        try:
            # turn init_image attachment data to pil image with mult of 64 width/height
            init_image = await url2pil(image_url)
            new_width = round(init_image.width/64)*64
            new_height = round(init_image.height/64)*64
            for i in range(5):
                if max(new_width, new_height) < 400:
                    new_width *= 2
                    new_height *= 2
                else:
                    break
            init_image = init_image.resize((new_width, new_height), Image.Resampling.BICUBIC)
            try:
                image_info = await pil2base64(init_image)
                source_image, width, height = image_info  
            except:
                await inter.response.send_message('Error, Image file does not have valid encoding', ephemeral=True)  
                return
        except:
            source_image = None
            print('Image load failed')
            await inter.response.send_message('Error, Invalid Image URL', ephemeral=True)  
            return
    else:
        source_image = None
        
    # set anime model if anime-ish prompt
    if 'anime' in prompt or 'waifu' in prompt:
        model = 'Anything v3'
        
    neg_prompt = '2d, grid, text'
    if not model in settings.model_list:
        await inter.response.send_message('Invalid Model: ```'+model+'```', ephemeral=True)
        return
    
    steps = 30
    cfg_scale = 8
    tiling = False
    
    
    
    #try:
    if 1:
        arguments = prompt.split('-')
        for i in range(len(arguments)):
            try:
                if 'http' in arguments[i]:
                    arguments.pop(i)
            except:
                break
        prompt = arguments.pop(0)
        if not len(arguments) == 0:  
            arguments = ('-' + '-'.join(arguments)).split(' ')
            print(arguments)
            for i, argument in enumerate(arguments):
                try:
                    keyvalue = arguments[i+1]
                except:
                    keyvalue = None
                print(f'{argument} : {keyvalue}')
                if argument in ['--ar', '-ar', '-aspect' ,'--aspect']:
                    width, height = await views.ar2size(keyvalue, settings.default_size)
                elif argument in ['--chaos', '-chaos']:
                    cfg_scale = int(keyvalue)//3.5
                elif argument in ['--no', '-no']:
                    neg_prompt = neg_prompt + ', ' + keyvalue
                elif argument in ['--quality', '-quality']:
                    steps = round(float(keyvalue)*35)
                elif argument in ['--niji', '-niji']:
                    model = 'Anything v3'
                elif argument in ['--tile', '-tile']:
                    tiling = True
                elif '-' in argument and len(argument) < 20:
                    await inter.response.send_message('Invalid argument: ```' + argument + '```')
        await inter.response.send_message('You got it, boss :saluting_face:', ephemeral=True)
    #except:
    #    await inter.response.send_message('Invalid Args', ephemeral=True)
    #    return
     
    karras = True
    dm = False
    imageurl = None
    userid = inter.author.mention

    try:
        nsfw = not inter.channel.nsfw or settings.nsfw_filter
            
    except AttributeError:
        dm = True
        nsfw = True
    
    vars = {
        'prompt': prompt,
        'neg_prompt': neg_prompt,
        'width': width,
        'height': height,
        'upscalers': 'GFPGAN',
        'model': model,
        'cfg_scale': cfg_scale,
        'imageurl': imageurl,
        'userid': userid,
        'channelid': inter.channel_id,
        'sampler': 'k_euler_a',
        'karras': karras,
        'steps': steps,
        'source_image': source_image,
        'source_processing': 'img2img',
        'source_mask': None,
        'filter':nsfw,
        'dm': dm,
        'author': inter.author,
        'denoising_strength': 0.50,
        'seed': -1,
        'inter_response': None,
        'amount': 4,
        'tiling' : tiling,
        'hires_fix' : True,
        'english' : True,
        'midj' : True,
        'raw_prompt' : raw_prompt,
        }
    global url
    global endpoint
    await query_api(settings.url + settings.endpoint, vars, inter)

@bot.slash_command(description='Generates images using Stable Horde img2img!')
async def riff(
    inter: disnake.ApplicationCommandInteraction,
    init_image: disnake.Attachment = commands.Param(description='Initial Image'),
    prompt: str = commands.Param(description='Command for the AI, e.g. \'Make her hair green\''),
    neg_prompt: str = commands.Param(default = '2D, grid, text', description='What the AI image model should avoid. Default: \'2D, grid, text\''),
    upscalers: str = commands.Param(choices=settings.processor_list, default='GFPGAN', description='Which Post-Processing to use for the images. Default: GFPGAN'),
    model: str = commands.Param(default=settings.default_riff,autocomplete=settings.autocomp_models, description='Which model to generate the image with. Default: ' + str(settings.default_riff)),
    cfg_scale: Optional[float] = commands.Param(default=8, le=30, ge=-40, description='How much the image should look like your prompt. Default: 8'),
    denoising_strength: int = commands.Param(default=50, name = 'image_guidance',le=100, ge=0, description = 'How much the image should match the provided, 100 = clone, 0 = no effect'),
    sampler: str = commands.Param(default = settings.default_sampler, description = 'ADVANCED: Which stable diffusion sampler to use. Default: ' + settings.default_sampler, choices=settings.sampler_list),
    steps: int = commands.Param(default=settings.default_steps, le=50, ge=1, description='Greater: Higher Image Quality but takes longer. Default: ' + str(settings.default_steps)),
    seed: int = commands.Param(default=-1, description='Seed for the image.'),
    amount: int = commands.Param(default=settings.default_images, choices = [1,2,4,6,8,9], description='Amount of images to generate. Default: ' + str(settings.default_images)),
    tiling: bool = commands.Param(default = False, description='Whether to have the image be repeating and tileable. Default: False'),
    hires_fix: bool = commands.Param(default = True, description='Improves Image Quality at high resolutions. Default: True'),
    english: bool = commands.Param(default = True, description='Set to false if prompt language is not English'),
    control_type: str = commands.param(default = None, choices=settings.acceptable_controls, description='ControlNet Parameter type'),
):
    raw_prompt = prompt
    if not model in settings.model_list and not model == 'pix2pix':
        await inter.response.send_message('Invalid Model: ```'+model+'```', ephemeral=True)
        return
        
    if not str(init_image.content_type) == init_image.content_type or not init_image.content_type.endswith(settings.input_types):
        await inter.response.send_message('Attachment is not valid image of types: WebP, PNG, JPG, JPEG', ephemeral=True)
        return
    
    print(f'Media Type: {init_image.content_type}')
    
    denoising_strength = 1-(denoising_strength/100)
    karras = True
    dm = False
    userid = inter.author.mention
    cfg_scale = round(cfg_scale * 2)/2
    
    # save init_image url for vars
    image_url = init_image.url
    
    # turn init_image attachment data to pil image with mult of 64 width/height
    init_image = await url2pil(image_url)
    new_width = round(init_image.width/64)*64
    new_height = round(init_image.height/64)*64
    for i in range(5):
        if max(new_width, new_height) < 400:
            new_width *= 2
            new_height *= 2
        else:
            break
    init_image = init_image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    try:
        image_info = await pil2base64(init_image)
        source_image, width, height = image_info  
    except:
        await inter.response.send_message('Error, Image file does not have valid encoding', ephemeral=True)  
        return
    
    try:
        nsfw = not inter.channel.nsfw or settings.nsfw_filter
    except AttributeError:
        dm = True
        nsfw = True
          
    
    
    embed=disnake.Embed(title='img2img generation queued with following parameters:', color=0xf0e000)
    embed.set_author(name='Your generation request is on its way!')
    embed.set_thumbnail(url=settings.embed_icon)
    embed.add_field(name='Prompt:', value=prompt[:512], inline=False)
    embed.add_field(name='Negative Prompt:', value=neg_prompt[:512], inline=False)
    embed.add_field(name='CFG Scale:', value=cfg_scale, inline=True)
    embed.add_field(name='Steps:', value=steps, inline=True)
    embed.add_field(name='Model:', value=model, inline=True)
    embed.add_field(name='Content Filter:', value=str(nsfw), inline=True)
    embed.add_field(name='Dimensions:', value=str(width) + 'x' + str(height), inline=True)
    print(nsfw)
    
    if settings.use_embeds:
        await inter.response.send_message(embed=embed) 
    else:
        user_response = 'Generating with following parameters:\nPrompt: ```' + prompt + '```Negative Prompt: ```' + neg_prompt + '```CFG Scale: ```' + str(cfg_scale) + '```Steps: ```' + str(steps) + '```for ' + userid + '. Please be patient'
        await inter.response.send_message(user_response) 
        
    inter_response = await inter.original_response() 
    
    vars = {
        'prompt': prompt,
        'neg_prompt': neg_prompt,
        'width': width,
        'height': height,
        'upscalers': upscalers,
        'model': model,
        'cfg_scale': cfg_scale,
        'imageurl': image_url,
        'userid': userid,
        'channelid': inter.channel_id,
        'sampler': sampler,
        'karras': karras,
        'steps': steps,
        'source_image': source_image,
        'source_processing': 'img2img',
        'source_mask': None,
        'filter':nsfw,
        'dm': dm,
        'author': inter.author,
        'denoising_strength': denoising_strength,
        'seed': seed,
        'inter_response': inter_response,
        'amount': amount,
        'tiling' : tiling,
        'hires_fix' : hires_fix,
        'english' : english,
        'control_type' : control_type,
        'midj' : False,
        'raw_prompt' : raw_prompt,
        }
    global url
    global endpoint
    await query_api(settings.url + settings.endpoint, vars, inter)

async def pil2base64(image):
    width, height = image.size
            
    rounded_width = 0
    rounded_height = 0

    if width <= height:
        rounded_width = settings.default_width
        rounded_height = (height/width*settings.default_width)
    else:
        rounded_height = settings.default_height
        rounded_width = (width/height*settings.default_width)
        
    rounded_width = round(rounded_width / 64) * 64
    rounded_height = round(rounded_height / 64) * 64
    rounded_width = min(max(rounded_width, 64), settings.max_size)
    rounded_height = min(max(rounded_height, 64), settings.max_size)
    image_resized = image.resize((rounded_width, rounded_height))
    bytesio = io.BytesIO()
    image_resized.save(bytesio, format = settings.format_type)
    
    bytesio.seek(0)
    image_string = base64.b64encode(bytesio.read()).decode()
    return [image_string, width, height]

async def url2base64(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            image_data = await response.read()
            image = Image.open(io.BytesIO(image_data))
            data = await pil2base64(image)
            return data
        
async def url2pil(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            image_data = await response.read()
            image = Image.open(io.BytesIO(image_data))
            return image


async def attach2base64(image):
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))
    data = await pil2base64(image)
    return data


async def shade_cells(width, height, gens, amount):
    rounded_width = rounded_height = 0

    if width <= height:
        rounded_width = 512
        rounded_height = (height/width*512)
    else:
        rounded_height = 512
        rounded_width = (width/height*512)
        
    width = round(rounded_width)
    height = round(rounded_height)
    
    dst = Image.new('RGB', (width*4, height*2))
    
    finished = gens['finished']
    processing = gens['processing']
    restarted = gens['restarted']
    waiting = gens['waiting']
    imagename = str(finished) + str(processing) + str(restarted) + str(waiting) + '_' + str(width) + 'x' + str(height) + 'n' + str(amount) + settings.img_type

    if os.path.isfile('waitcache/' + imagename):
        return 'waitcache/' + imagename
    generations = []
    for i in range(finished):
        generations.append(settings.finished_color)
        
    for i in range(restarted):
        generations.append(settings.restarted_color)
        
    for i in range(waiting):
        generations.append(settings.waiting_color)

    for i in range(processing):
        generations.append(settings.processing_color)
    
    for i in range(amount-len(generations)):
        generations.append(settings.error_color)

    images = []
    for color in generations:
        images.append(Image.new('RGB', (width, height), color=color))

    dst = await views.make_grid(images, amount)
    dst.save('waitcache/' + imagename)
    return 'waitcache/' + imagename

async def fetch_image(session, url):
    async with session.get(url) as response:
        return await response.read()

async def url_to_pil(url):
    async with aiohttp.ClientSession() as session:
        image_data = await fetch_image(session, url)
        with io.BytesIO(image_data) as f:
            with Image.open(f) as img:
                return img


async def query_api(url, vars, interaction):
    if vars['midj']:
        prompt = await settings.enhance_prompt(vars['prompt'])
    else:
        prompt = vars['prompt']
     # temporary, remove later
    if randint(1,20) == 1:
        settings.autocomp_models = settings.redefine_autocomp()
    msg_channel = bot.get_channel(vars['channelid'])
    if vars['dm']:
        msg_channel = vars['author']
    try:
        checkprompt = intents.check_cp(vars['prompt'])
    
        if checkprompt[0]:
            await msg_channel.send(vars['userid'] +', Suspected Child Porn prompt has been blocked and reported. If repeated multiple times, risk being banned.')
            return
    except:
        print('WARNING: Unfiltered Prompt')
    try:
        amount = vars['amount']
    except:
        amount = settings.default_images
        
    if vars['model'] == 'Midjourney Diffusion' and not vars['prompt'].startswith('midjrny-v4 style'):
        await msg_channel.send('Auto-adding \'midjrny-v4 style\' to start of prompt!')
        vars['prompt'] = 'midjrny-v4 style ' + vars['prompt']
    
    headers = {
    # Already added when you pass json= but not when you pass data=
    # 'Content-Type': 'application/json',
        'apikey': settings.sd_api_key,
    }

    params = {
        'sampler_name': vars['sampler'],
        #'seed_variation': 1,
        'cfg_scale': 7.5,
        'height': vars['height'],
        'width': vars['width'],
        'post_processing': [
            vars['upscalers']
            ],
        'karras': vars['karras'],
        'tiling': vars['tiling'],
        'hires_fix': vars['hires_fix'],
        'steps': vars['steps'],
        'n': amount,
        'denoising_strength': vars['denoising_strength'],
        }
    
    if 'control_type' in vars:
        if not vars['control_type'] == None:
            params.update({'control_type': vars['control_type']})

    if not vars.get('seed', -1) == -1:
        params.update({'seed': str(vars.get('seed', -1))})

    if not vars['english']:
        vars['prompt'] = await views.trans_query(vars['prompt'])
        vars['neg_prompt'] = await views.trans_query(vars['neg_prompt'])

    
        
    json_data = {
        'prompt': prompt + '###' + vars['neg_prompt'],
        'params': params,
        'nsfw': True,
        'trusted_workers': True,
        'censor_nsfw': False,
        'models': [
            vars['model']
            ],
        'source_processing': vars['source_processing'],
          }
    
    if vars['upscalers'] == 'RealESRGAN_x4plus':
        vars['width'] *= 4
        vars['height'] *= 4
    
    if not vars['source_image'] == None:
        json_data.update({'source_image': vars['source_image']})
        
    if not vars['source_mask'] == None:
        json_data.update({'source_mask': vars['source_mask']})
    
    vars.pop('source_image')
    vars.pop('source_mask')    
    
    vars['midj'] = True
    # Initialize a variable to track whether the desired status has been received, failed, and a list to capture results
    done = False
    failed = False
    results = []
    message = await msg_channel.send('Querying job for user ' + vars['userid'])
    if not settings.accept_dm:
        await message.edit('Sorry, but generating in DM\'s is disabled')
        return
    # Keep querying the API until the desired status is received
    async with aiohttp.ClientSession() as session:
        async with session.post(url + 'async', json=json_data, headers=headers) as response:
            codeid = await response.json()
            if not 'id' in codeid:
                await msg_channel.send('Error: Job for user ' + vars['userid'] + ' failed. Full Error Report: ' + str(codeid))
                return
            codeid = codeid['id']
            print(codeid)
            
        start = time.time()
        while not done:
            async with session.get(url + 'check/' + codeid) as response:
                try:
                    rawtext = await response.text()
                except:
                    await message.edit('Server timed out')
                print(codeid + '  :  ' + rawtext)
                rstatus = response.status
                print(type(rstatus))
                if rstatus == 200 and hasattr(response, 'json'):
                    gens = await response.json()
                    if 'finished' in gens:
                        done = gens['done']
                        
                        filelocation = await shade_cells(vars['width'], vars['height'], gens, vars['amount'])
                        
                        color = await views.average_color(filelocation)
                        
                        embed=disnake.Embed(title='Generating job!', color=color)
                        embed.add_field(name='Finished:', value=str(gens['finished']), inline=True)
                        embed.add_field(name='Processing:', value=str(gens['processing']), inline=True)
                        if int(gens['restarted'])>0:
                            embed.add_field(name='Restarted:', value=str(gens['restarted']), inline=True)
                        embed.add_field(name='Waiting', value=str(gens['waiting']), inline=True)
                        embed.add_field(name='Content Filter:', value=str(vars['filter']), inline=True)
                        embed.add_field(name='Code(for debugging):', value='||' + codeid + '||', inline=True)
                        if int(gens['queue_position']) > 1:
                            embed.add_field(name='Queue Position:', value=str(gens['queue_position']), inline=True)
                        embed.set_image(file=disnake.File(filelocation))
                        
                        if not vars['midj']:
                            await message.edit(embed=embed)
                        else:
                            await message.edit(f'**{vars["raw_prompt"]}** - ' + vars['userid'], file=disnake.File(filelocation))
                        
                        current = time.time()-start
                        
                        if current >= settings.timeout:
                            done = True
                            
                    else:
                        await msg_channel.send('If you are seeing this message, the code is outdated or something serious is wrong.')
                        return
                else:
                    await msg_channel.send('Error: Job for user ' + vars['userid'] + ' failed. Status Code ' + str(rstatus) + ' received. Full Error Report: ' + rawtext)
                    await session.close()
                    return
            # Wait for the specified amount of time before making the next query
            if not done:
                await asyncio.sleep(settings.wait_time)
            print(failed)
            print(done)
        await views.add_data(codeid + '_' + str(vars['amount']))
        async with session.get(url + 'status/' + codeid) as response:
            rawtext = await response.text()
            rstatus = response.status
            print(type(rstatus))
            
            udata = {}
            
            filtered_images = 0
            filtered_list = []
            if rstatus == 200 and hasattr(response, 'json'):
                rawjson = await response.json()
                rawjson = rawjson['generations']
                for counter, image in enumerate(rawjson): 
                    imgname = 'nsfwcache/' + codeid + '_' + str(counter) +'.jpeg'
                    imgdata = image['img']
                                        
                    img = await url_to_pil(imgdata)
                    vars['width'] = img.width
                    vars['height'] = img.height
                    
                    try:
                        udata.update({str(counter):{'model':image['model'],'seed':image['seed']}})
                    except:
                        udata.update({str(counter):{'model':'Unknown', 'seed':0}})
                    
                    if vars['filter']:
                        img.save(imgname)
                        nsfwdata = predict.classify(model, imgname)
                        if nsfwdata[imgname]['neutral'] + nsfwdata[imgname]['drawings'] >= settings.filter_strength:
                            print('Check clear!')
                            os.remove(imgname)
                        else:
                            width, height = img.size
                            print('nsfw filtered')
                            filtered_images += 1
                            filtered_list.append(counter)
                            img = Image.new(mode = 'RGB', size = (width, height),color = settings.nsfw_color)
                            if not settings.save_nsfw:
                                os.remove(imgname)
                        
                    results.append(img)
                
                errors = vars['amount']-len(results)
                error_image = Image.new(mode='RGB', size=(vars['width'], vars['height']), color = settings.error_color)
                
                author = vars.pop('author')
                print(author.id)
                vars.update({'author': author.id})
                udata.update(vars)
                
                udata.pop('inter_response')
                
                with open('textcache/' + codeid + '.txt', 'w') as f:
                    f.write(str(udata)) # .replace('\'', '"')
                                
                for i in range(vars['amount']-len(results)):
                    results.append(error_image)
                finalgrid = await views.make_grid(results, vars['amount'])
                finalgrid.save('imagecache/' + codeid + settings.img_type)
                
                if settings.use_embeds:
                    
                    view = GenView()
                    view = view.make_view(vars['amount'], codeid, vars['midj'])
                        
                    filepath = 'imagecache/' + codeid + settings.img_type
                    
                    color = await views.average_color(filepath)
                                        
                    embed=disnake.Embed(title='Job Complete!', color=color)
                    embed.add_field(name='Content Filter:', value=str(vars['filter']), inline=True)
                    embed.add_field(name='Images timed out:', value=str(errors), inline=True)
                    embed.add_field(name='Filtered images:', value=str(filtered_images), inline=True)
                    embed.add_field(name='Code(for debugging):', value='||' + codeid + '||', inline=True)
                    await message.delete()
                    if amount == 1:
                        await upscale(interaction, codeid, 200)
                        return
                    if not vars['midj']:
                        try:
                            message = await vars['inter_response'].reply('Job done for user ' + vars['userid'], embed=embed, view=view)
                            embed.set_image(file=disnake.File(filepath))
                            await message.edit('Job done for user ' + vars['userid'], embed=embed, view=view)
                        except:
                            message = await msg_channel.send('Job done for user ' + vars['userid'], embed=embed, view=view)
                            embed.set_image(file=disnake.File(filepath))
                            await message.edit('Job done for user ' + vars['userid'], embed=embed, view=view)
                            #try:
                            #    await msg_channel.send('Job done for user ' + vars['userid'], embed=embed, view=view)
                            #except:
                            #    await msg_channel.send_message('Job done for user ' + vars['userid'], embed=embed, view=view)
                        with open('textcache/' + codeid + '-url.txt', 'w') as f:
                            f.write(message.embeds[0].image.url)
                        os.remove(filepath)
                    else:
                        finished_message = await msg_channel.send(f'**{vars["raw_prompt"]}** - ' + vars['userid'], file=disnake.File(filepath), view=view)
                        with open('textcache/' + codeid + '-url.txt', 'w') as f:
                            f.write(finished_message.attachments[0].url)
                        os.remove(filepath)
                else:
                    await message.edit(file=disnake.File('imagecache/' + codeid + settings.img_type))

            elif settings.use_embeds:
                embed=disnake.Embed(title='Job Failed', color=0xff0000)
                embed.add_field(name='Content Filter:', value=str(vars['nsfw']), inline=True)
                embed.add_field(name='Code(for debugging):', value='||' + codeid + '||', inline=True)
                await message.edit('Job done for ' + vars['userid'] + '!', embed=embed)
            else:
                await message.edit('Job Failed')
    await session.close()
    # Jesus christ thank god its finally over
    return        
    
bot.run(settings.token)