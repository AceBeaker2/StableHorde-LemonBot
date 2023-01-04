import disnake
from typing import Optional
from disnake.ext import commands
from disnake.ui import Button
from config import *
#import nest_asyncio
# above line is weird spyder workaround. DW about it.
import asyncio
import aiohttp
import time
import logging
from PIL import Image, ImageDraw
from nsfw_detector import predict
import base64
import io
from urllib.parse import urlparse
import ast

import os
import json

model = predict.load_model('./checker.h5')
# gets model

logging.basicConfig(level=logging.INFO)
#nest_asyncio.apply()
# weird spyder thing I don't get it

bot = commands.Bot(
    command_prefix=disnake.ext.commands.when_mentioned,
    )

@bot.event
async def on_ready():
    print('The bot is ready!')

@bot.slash_command(description='Help menu!!!')
async def config(inter):
    embed=disnake.Embed(title='Help menu!', color=0x00ff40)
    embed.set_image(url='https://i.imgur.com/QLT48xE.png')
    
    await inter.response.send_message(embed=embed)
    
@bot.slash_command(description='Alternate upscale endpoint if the buttons don\'t work')
async def upscale(
    inter: disnake.ApplicationCommandInteraction,
    code: str = commands.Param(description = 'Code for upscaling', min_length=36, max_length=36),
    number: int = commands.Param(le=8, ge=1,description = 'Number to upscale'),
    ):
    if not os.path.isfile('imagecache/' + code + img_type):
        await inter.response.send_message('Image not found', ephemeral=True)
        return

    await inter.response.defer(with_message=True)

    number -= 1    
    filepath = await upscale_code(code, number)
    
    with open('textcache/' + code + '.txt', 'r') as f:
        text = f.read()
        vars = ast.literal_eval(text)
    
    #try:
    color = await average_color(filepath)
    embed=disnake.Embed(title='Upscale Done!', color=color)
    embed.add_field(name='Seed:', value=vars[str(number)]['seed'], inline=False)
    embed.set_image(file=disnake.File(filepath))

    view = disnake.ui.View(timeout=None)

    moreinfo = disnake.ui.Button(custom_id='info', label='More Info', style=disnake.ButtonStyle.blurple, emoji='🤔')
    variate = disnake.ui.Button(custom_id='variate', label='Make Variations', style=disnake.ButtonStyle.green)
    outpaint = disnake.ui.Button(custom_id='outpaint', label='Outpaint', style=disnake.ButtonStyle.green)
    
    #except:
    #    embed = disnake.Embed('Upscale Failed!', color=0xff0000)
    #    
    #    view = disnake.ui.View()
    
    async def infocallback(interaction):
        embed=disnake.Embed(title='Status Sheet', color=color)
        
        embed.add_field(name='Seed:', value=vars[str(number)]['seed'], inline=False)
        embed.add_field(name='Sampler:', value=vars['sampler'], inline=False)
        embed.add_field(name='Prompt:', value=vars['prompt'][:1024], inline=False)
        embed.add_field(name='Negative Prompt:', value=vars['neg_prompt'][:1024], inline=False)
        embed.add_field(name='Model:', value=vars['model'], inline=False)
        embed.add_field(name='Content Filter', value=str(vars['filter']), inline=False)
        embed.add_field(name='CFG Scale:', value=vars['cfg_scale'], inline=False)
        embed.add_field(name='Steps:', value=str(vars['steps']), inline=False)
        embed.add_field(name='Dimensions:', value=str(vars['width']) + 'x' + str(vars['height']), inline=False)
        embed.add_field(name='Upscaler:', value=str(vars['upscalers']), inline=False)
        embed.add_field(name='Author:', value='<@'+str(vars['author'])+'>', inline=False)
        
        embed.set_image(file=disnake.File(filepath))
        await interaction.send(embed=embed)
    
    async def variatecallback(interaction):
        await interaction.send('Your Variation is on the way!', ephemeral=True) 
        dm = False
        try:
             nsfw = not interaction.channel.nsfw or nsfw_filter
        except AttributeError:
             dm = True
             nsfw = True
               
        image_info = await url2base64(imageurl)
        source_image, width, height = image_info

        author = interaction.user

        inter_response = await inter.original_response()         
        
        vvars = {
             'prompt': vars['prompt'],
             'neg_prompt': vars['neg_prompt'],
             'width': vars['width'],
             'height': vars['height'],
             'upscalers': 'GFPGAN',
             'model': vars['model'],
             'cfg_scale': vars['cfg_scale'],
             'imageurl': imageurl,
             'userid': author.mention,
             'channelid': inter.channel_id,
             'sampler': vars['sampler'],
             'karras': True,
             'steps': 10,
             'source_image': source_image,
             'source_processing': 'img2img',
             'source_mask': None,
             'filter':nsfw,
             'dm': dm,
             'author': author,
             'denoising_strength': 0.8,
             'seed': -1,
             'inter_response': inter_response,
             }
        await query_api(url + endpoint, vvars)
    
    async def outpaintcallback(interaction):
        author = interaction.user
        dm = False
        try:
            nsfw = not interaction.channel.nsfw or nsfw_filter
        except AttributeError:
            dm = True
            nsfw = True
              
        try:
            image_info = await url2mask(imageurl)
            source_image, mask_string, width, height = image_info  
        except:
            await inter.response.send_message('Error, URL invalid. Please try another image', ephemeral=True)  
            return
        
        await interaction.response.send_modal(
        title='Outpaint image',
        custom_id='outpaint_modal',
        components=[
                disnake.ui.TextInput(
                label='Prompt',
                placeholder='The prompt for outpainting',
                custom_id='prompt',
                style=disnake.TextInputStyle.paragraph,
                max_length=1024,
                ),
                disnake.ui.TextInput(
                label='Negative Prompt',
                placeholder='The negative prompt for outpainting',
                custom_id='neg_prompt',
                required=False,
                value='2D, grid, text',
                style=disnake.TextInputStyle.paragraph,
                max_length=1024,
                ),
            ],
        )
        try:
            modal_inter: disnake.ModalInteraction = await bot.wait_for(
                'modal_submit',
                check=lambda i: i.custom_id == 'outpaint_modal' and i.author.id == interaction.author.id,
                timeout=600,
                )
        except asyncio.TimeoutError:
            # The user didn't submit the modal in the specified period of time.
            # This is done since Discord doesn't dispatch any event for when a modal is closed/dismissed.
            return
        await modal_inter.response.send_message('Outpaint request is on its way!')
        
        inter_response = await modal_inter.original_response() 
        
        ovars = {
            'prompt': modal_inter.text_values['prompt'],
            'neg_prompt': modal_inter.text_values['neg_prompt'],
            'width': width,
            'height': height,
            'upscalers': 'GFPGAN',
            'model': 'stable_diffusion_inpainting',
            'cfg_scale': vars['cfg_scale'],
            'imageurl': imageurl,
            'userid': author.mention,
            'channelid': inter.channel_id,
            'sampler': 'k_euler_a',
            'karras': True,
            'steps': 10,
            'source_image': source_image,
            'source_processing': 'inpainting',
            'source_mask': mask_string,
            'filter':nsfw,
            'dm': dm,
            'author': inter.author,
            'denoising_strength': 0.75,
            'seed': -1,
            'inter_response': inter_response,
            }
        global url
        global endpoint
        await query_api(url + endpoint, ovars)
    
    outpaint.callback = outpaintcallback
    moreinfo.callback = infocallback
    variate.callback = variatecallback
    
    view.add_item(variate)
    view.add_item(outpaint)
    view.add_item(moreinfo)
    
    await inter.edit_original_message(embed=embed, view=view) 
    message = await inter.original_message()
    imageurl = message.embeds[0].image.url
    
@bot.slash_command(description='Generates images using Stable Horde!')
async def generate(
    inter: disnake.ApplicationCommandInteraction,
    prompt: str = commands.Param(description='What the AI-generated image should be of.'),
    neg_prompt: str = commands.Param(default = '2D, grid, text', description='What the AI image model should avoid. Default: \'2D, grid, text\''),
    upscalers: str = commands.Param(choices=processor_list, default='GFPGAN', description='Which Post-Processing to use for the images. Default: GFPGAN'),
    model: str = commands.Param(default=default_model,choices=model_list, description='Which model to generate the image with. Default: ' + str(default_model)),
    cfg_scale: Optional[float] = commands.Param(default=8, le=30, ge=-40, description='How much the image should look like your prompt. Default: 8'),
    #imageurl: str = commands.Param(name = 'init_image', default = None, description='Initial image for img2img.'),
    width: int = commands.Param(default = default_width, le=1024, ge=64, description='Width of the final image. Default: ' + str(default_width)),
    height: int = commands.Param(default = default_height, le=1024, ge=64, description='Height of the final image. Default: ' + str(default_height)),    
    sampler: str = commands.Param(default = default_sampler, description = 'ADVANCED: Which stable diffusion sampler to use. Default: ' + default_sampler, choices=sampler_list),
    steps: int = commands.Param(default=default_steps, le=50, ge=1, description='Greater: Higher Image Quality but takes longer. Default: ' + str(default_steps)),
    seed: int = commands.Param(default=-1, description='Seed for the image.')
): 
    karras = True
    dm = False
    imageurl = None
    userid = inter.author.mention
    cfg_scale = round(cfg_scale * 2)/2
    width = round(width/64)*64
    height = round(height/64)*64
    try:
        nsfw = not inter.channel.nsfw or nsfw_filter
            
    except AttributeError:
        dm = True
        nsfw = True
    
    embed=disnake.Embed(title='Generation queued with following parameters:', color=0xf0e000)
    embed.set_author(name='Your generation request is on its way!')
    embed.set_thumbnail(url=embed_icon)
    embed.add_field(name='Prompt:', value=prompt[:512], inline=False)
    embed.add_field(name='Negative Prompt:', value=neg_prompt[:512], inline=False)
    embed.add_field(name='CFG Scale:', value=cfg_scale, inline=True)
    embed.add_field(name='Steps:', value=steps, inline=True)
    embed.add_field(name='Model:', value=model, inline=True)
    embed.add_field(name='Content Filter:', value=str(nsfw), inline=True)
    embed.add_field(name='Dimensions:', value=str(width) + 'x' + str(height), inline=True)
    
    print(nsfw)
    
    if use_embeds:
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
        }
    global url
    global endpoint
    await query_api(url + endpoint, vars)


@bot.slash_command(description='Generates images using Stable Horde img2img!')
async def riff(
    inter: disnake.ApplicationCommandInteraction,
    prompt: str = commands.Param(description='What the AI-generated image should be of.'),
    neg_prompt: str = commands.Param(default = '2D, grid, text', description='What the AI image model should avoid. Default: \'2D, grid, text\''),
    upscalers: str = commands.Param(choices=processor_list, default='GFPGAN', description='Which Post-Processing to use for the images. Default: GFPGAN'),
    model: str = commands.Param(default=default_model,choices=model_list, description='Which model to generate the image with. Default: ' + str(default_model)),
    cfg_scale: Optional[float] = commands.Param(default=8, le=30, ge=-40, description='How much the image should look like your prompt. Default: 8'),
    imageurl: str = commands.Param(name = 'init_image', description='Initial image for img2img.'),
    denoising_strength: int = commands.Param(name = 'image_guidance',le=100, ge=0, description = 'How much the image should match the provided, 100 = clone, 0 = no effect'),
    sampler: str = commands.Param(default = default_sampler, description = 'ADVANCED: Which stable diffusion sampler to use. Default: ' + default_sampler, choices=sampler_list),
    steps: int = commands.Param(default=default_steps, le=50, ge=1, description='Greater: Higher Image Quality but takes longer. Default: ' + str(default_steps)),
    seed: int = commands.Param(default=-1, description='Seed for the image.'),
):
    denoising_strength = 1-denoising_strength/105
    karras = True
    dm = False
    userid = inter.author.mention
    cfg_scale = round(cfg_scale * 2)/2
    try:
        nsfw = not inter.channel.nsfw or nsfw_filter
    except AttributeError:
        dm = True
        nsfw = True
    
    if not urlparse(imageurl).netloc == 'cdn.discordapp.com' or not urlparse(imageurl).path.endswith(input_types):
        await inter.response.send_message('URL is either not cdn.discordapp, or image type is not valid', ephemeral=True)
        return
          
    try:
        image_info = await url2base64(imageurl)
        source_image, width, height = image_info  
    except:
        await inter.response.send_message('Error, URL invalid. Please try another image', ephemeral=True)  
        return
    
    embed=disnake.Embed(title='img2img generation queued with following parameters:', color=0xf0e000)
    embed.set_author(name='Your generation request is on its way!')
    embed.set_thumbnail(url=embed_icon)
    embed.add_field(name='Prompt:', value=prompt[:512], inline=False)
    embed.add_field(name='Negative Prompt:', value=neg_prompt[:512], inline=False)
    embed.add_field(name='CFG Scale:', value=cfg_scale, inline=True)
    embed.add_field(name='Steps:', value=steps, inline=True)
    embed.add_field(name='Model:', value=model, inline=True)
    embed.add_field(name='Content Filter:', value=str(nsfw), inline=True)
    embed.add_field(name='Dimensions:', value=str(width) + 'x' + str(height), inline=True)
    print(nsfw)
    
    if use_embeds:
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
        'source_image': source_image,
        'source_processing': 'img2img',
        'source_mask': None,
        'filter':nsfw,
        'dm': dm,
        'author': inter.author,
        'denoising_strength': denoising_strength,
        'seed': seed,
        'inter_response': inter_response,
        }
    global url
    global endpoint
    await query_api(url + endpoint, vars)

@bot.slash_command(description='Outpaints images using Stable Horde outpainting!')
async def outpaint(
    inter: disnake.ApplicationCommandInteraction,
    prompt: str = commands.Param(description='What the AI-generated image should be of.'),
    neg_prompt: str = commands.Param(default = '2D, grid, text', description='What the AI image model should avoid. Default: \'2D, grid, text\''),
    upscalers: str = commands.Param(choices=processor_list, default='GFPGAN', description='Which Post-Processing to use for the images. Default: GFPGAN'),
    cfg_scale: Optional[float] = commands.Param(default=8, le=30, ge=-40, description='How much the image should look like your prompt. Default: 8'),
    imageurl: str = commands.Param(name = 'init_image', description='Initial image for img2img.'),
    sampler: str = commands.Param(default = default_sampler, description = 'ADVANCED: Which stable diffusion sampler to use. Default: ' + default_sampler, choices=sampler_list),
    steps: int = commands.Param(default=default_steps, le=50, ge=1, description='Greater: Higher Image Quality but takes longer. Default: ' + str(default_steps)),
    seed: int = commands.Param(default=-1, description='Seed for the image.'),
):
    model = 'stable_diffusion_inpainting'
    
    karras = True
    dm = False
    userid = inter.author.mention
    cfg_scale = round(cfg_scale * 2)/2
    try:
        nsfw = not inter.channel.nsfw or nsfw_filter
    except AttributeError:
        dm = True
        nsfw = True
    
    if not urlparse(imageurl).netloc == 'cdn.discordapp.com' or not urlparse(imageurl).path.endswith(input_types):
        await inter.response.send_message('URL is either not cdn.discordapp, or image type is not valid', ephemeral=True)
        return
          
    try:
        image_info = await url2mask(imageurl)
        source_image, mask_string, width, height = image_info  
    except:
        await inter.response.send_message('Error, URL invalid. Please try another image', ephemeral=True)  
        return

    
    embed=disnake.Embed(title='Outpainting generation queued with following parameters:', color=0xf0e000)
    embed.set_author(name='Your generation request is on its way!')
    embed.set_thumbnail(url=embed_icon)
    embed.add_field(name='Prompt:', value=prompt[:512], inline=False)
    embed.add_field(name='Negative Prompt:', value=neg_prompt[:512], inline=False)
    embed.add_field(name='CFG Scale:', value=cfg_scale, inline=True)
    embed.add_field(name='Steps:', value=steps, inline=True)
    embed.add_field(name='Model:', value=model, inline=True)
    embed.add_field(name='Content Filter:', value=str(nsfw), inline=True)
    embed.add_field(name='Dimensions:', value=str(width) + 'x' + str(height), inline=True)
    print(nsfw)
    
    if use_embeds:
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
        'source_image': source_image,
        'source_processing': 'inpainting',
        'source_mask': mask_string,
        'filter':nsfw,
        'dm': dm,
        'author': inter.author,
        'denoising_strength': 0.75,
        'seed': seed,
        'inter_response': inter_response,
        }
    global url
    global endpoint
    await query_api(url + endpoint, vars)

async def make_grid(images):
    im0 = images[0]
    dst = Image.new('RGB', (im0.width*4, im0.height*2))
    dst.paste(images[0], (0,0))
    dst.paste(images[1], (im0.width, 0))
    dst.paste(images[2], (im0.width*2,0))
    dst.paste(images[3], (im0.width*3, 0))
    dst.paste(images[4], (0,im0.height))
    dst.paste(images[5], (im0.width, im0.height))
    dst.paste(images[6], (im0.width*2,im0.height))
    dst.paste(images[7], (im0.width*3, im0.height))
    return dst

async def url2base64(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            image_data = await response.read()
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            
            rounded_width = 0
            rounded_height = 0

            if width <= height:
                rounded_width = default_width
                rounded_height = (height/width*default_width)
            else:
                rounded_height = default_height
                rounded_width = (width/height*default_width)

            rounded_width = round(rounded_width / 64) * 64
            rounded_height = round(rounded_height / 64) * 64
            rounded_width = min(max(rounded_width, 64), 1024)
            rounded_height = min(max(rounded_height, 64), 1024)
            image_resized = image.resize((rounded_width, rounded_height))
            bytesio = io.BytesIO()
            image_resized.save(bytesio, format = format_type)
            
            bytesio.seek(0)
            image_string = base64.b64encode(bytesio.read()).decode()
            return [image_string, rounded_width, rounded_height]
        
async def url2mask(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            image_data = await response.read()
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            
            rounded_width = 0
            rounded_height = 0

            if width <= height:
                rounded_width = 512
                rounded_height = (height/width*512)
            else:
                rounded_height = 512
                rounded_width = (width/height*512)
            
            rounded_width = round(width / 64) * 64
            rounded_height = round(height / 64) * 64
            rounded_width = min(max(rounded_width, 64), 1024)
            rounded_height = min(max(rounded_height, 64), 1024)
            
            image_resized = image.resize((rounded_width//3+2, rounded_height//3+2))
            
            canvas = Image.new('RGB', (rounded_width, rounded_height), color='white')
            
            xy = (rounded_width//3)-1
            
            canvas.paste(image_resized, (xy, xy))
            
            bytesio = io.BytesIO()
            canvas.save(bytesio, format = format_type)
            bytesio.seek(0)
            image_string = base64.b64encode(bytesio.read()).decode()
            
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((xy, xy, xy*2, xy*2), fill='black')
            
            bytesio = io.BytesIO()
            canvas.save(bytesio, format = format_type)
            bytesio.seek(0)
            mask_string = base64.b64encode(bytesio.read()).decode()
            
            return [image_string, mask_string, rounded_width, rounded_height]

async def average_color(path):
    im = Image.open(path)
    
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

async def shade_cells(width, height, gens):
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
    draw = ImageDraw.Draw(dst)
    
    finished = gens['finished']
    processing = gens['processing']
    restarted = gens['restarted']
    waiting = gens['waiting']
    imagename = str(finished) + str(processing) + str(restarted) + str(waiting) + '_' + str(width) + 'x' + str(height) + img_type

    if os.path.isfile('waitcache/' + imagename):
        return 'waitcache/' + imagename
    generations = []
    for i in range(finished):
        generations.append(finished_color)
        
    for i in range(restarted):
        generations.append(restarted_color)
        
    for i in range(waiting):
        generations.append(waiting_color)

    for i in range(processing):
        generations.append(processing_color)
    
    for i in range(8):
        generations.append(error_color)

    for number in range(8):
        if number > 3:
            top = dst.height/2
        else:
            top = 0
        bottom = top + (dst.height/2)
        
        left = (dst.width/4) * (number % 4)
        right = left + (dst.width/4)
        print(left, top, right, bottom)
        shape = [(left, top), (right, bottom)]
        draw.rectangle(shape, fill=generations[number])
    dst.save('waitcache/' + imagename)
    return 'waitcache/' + imagename
    
async def upscale_code(code, number):
    filepath = 'upscalecache/' + code + '_' + str(number) + img_type
    if not os.path.isfile(filepath):
        img = Image.open('imagecache/' + code + img_type)    
        if number > 3:
            top = img.height/2
        else:
            top = 0
            
        bottom = top + (img.height/2)
            
        left = (img.width/4) * (number % 4)
        right = left + (img.width/4)
        print(left, top, right, bottom)
        img = img.crop((left, top, right, bottom))
    
        img.save(filepath)
    return filepath


async def query_api(url, vars):
    global wait_time
    
    msg_channel = bot.get_channel(vars['channelid'])
    if vars['dm']:
        msg_channel = vars['author']
        
    if vars['model'] == 'Midjourney Diffusion' and not vars['prompt'].startswith('midjrny-v4 style'):
        await msg_channel.send('Auto-adding \'midjrny-v4 style\' to start of prompt!')
        vars['prompt'] = 'midjrny-v4 style ' + vars['prompt']
    
    headers = {
    # Already added when you pass json= but not when you pass data=
    # 'Content-Type': 'application/json',
        'apikey': sd_api_key,
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
        'steps': vars['steps'],
        'n': 8,
        'denoising_strength': vars['denoising_strength'],
        }

    if not vars['seed'] == -1:
        params.update({'seed': str(vars['seed'])})

    json_data = {
        'prompt': vars['prompt'] + '###' + vars['neg_prompt'],
        'params': params,
        'nsfw': not(vars['filter']),
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
    
    # Initialize a variable to track whether the desired status has been received, failed, and a list to capture results
    done = False
    failed = False
    results = []
    message = await msg_channel.send('Querying job for user ' + vars['userid'])
    if not accept_dm:
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
                        
                        filelocation = await shade_cells(vars['width'], vars['height'], gens)
                        
                        color = await average_color(filelocation)
                        
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
                        
                        if use_embeds:
                            await message.edit(embed=embed)
                        else:
                            await message.edit('Finished: ' + str(gens['finished']) + ', Processing: ' + str(gens['processing']) + ', Restarted: ' + str(gens['restarted']) + ', Waiting: ' + str(gens['waiting']) + ' Content filter: ' + str(vars['nsfw']) + ' Code(for debugging): ||' + codeid + '||', file=disnake.File(filelocation))
                        
                        current = time.time()-start
                        
                        if current >= timeout:
                            done = True
                            
                    else:
                        await msg_channel.send('If you are seeing this message, the code is outdated or something serious is wrong.')
                        return
                else:
                    await msg_channel.send('Error: Job for user ' + vars['userid'] + ' failed. Status Code ' + str(rstatus) + ' received. Full Error Report: ' + rawtext)
                    await session.close()
                    return
            # Wait for the specified amount of time before making the next query
            await asyncio.sleep(wait_time)
            print(failed)
            print(done)
            
        async with session.get(url + 'status/' + codeid) as response:
            rawtext = await response.text()
            rstatus = response.status
            print(type(rstatus))
            counter = 0
            
            udata = {}
            
            filtered_images = 0
            filtered_list = []
            if rstatus == 200 and hasattr(response, 'json'):
                
                rawjson = await response.json()
                rawjson = rawjson['generations']
                for image in rawjson: 
                    imgdata = base64.b64decode(image['img'])
                    
                    try:
                        udata.update({str(counter):{'model':image['model'],'seed':image['seed']}})
                    except:
                        udata.update({str(counter):{'model':'Unknown', 'seed':0}})

                    img = Image.open(io.BytesIO(imgdata))
                    if vars['filter']:
                        imgname = 'nsfwcache/' + codeid + '_' + str(counter) +'.jpeg'
                        img.save(imgname)
                        nsfwdata = predict.classify(model, imgname)
                        if nsfwdata[imgname]['neutral'] + nsfwdata[imgname]['drawings'] >= filter_strength:
                            print('Check clear!')
                            os.remove(imgname)
                        else:
                            width, height = img.size
                            print('nsfw filtered')
                            filtered_images += 1
                            filtered_list.append(counter)
                            img = Image.new(mode = 'RGB', size = (width, height),color = nsfw_color)
                            if not save_nsfw:
                                os.remove(imgname)
                        
                    results.append(img)
                    counter += 1
                
                errors = 8-len(results)
                error_image = Image.new(mode='RGB', size=(vars['width'], vars['height']), color = error_color)
                
                author = vars.pop('author')
                print(author.id)
                vars.update({'author': author.id})
                udata.update(vars)
                
                udata.pop('inter_response')
                
                with open('textcache/' + codeid + '.txt', 'w') as f:
                    f.write(str(udata)) # .replace('\'', '"')
                                
                for i in range(8):
                    results.append(error_image)
                finalgrid = await make_grid(results)
                finalgrid.save('imagecache/' + codeid + img_type)
                
                if use_embeds:
                    u1 = disnake.ui.Button(custom_id='u1', label='U1', style=disnake.ButtonStyle.blurple, row=0)
                    u2 = disnake.ui.Button(custom_id='u2', label='U2', style=disnake.ButtonStyle.blurple, row=0)
                    u3 = disnake.ui.Button(custom_id='u3', label='U3', style=disnake.ButtonStyle.blurple, row=0)
                    u4 = disnake.ui.Button(custom_id='u4', label='U4', style=disnake.ButtonStyle.blurple, row=0)
                    u5 = disnake.ui.Button(custom_id='u5', label='U5', style=disnake.ButtonStyle.blurple, row=1)
                    u6 = disnake.ui.Button(custom_id='u6', label='U6', style=disnake.ButtonStyle.blurple, row=1)
                    u7 = disnake.ui.Button(custom_id='u7', label='U7', style=disnake.ButtonStyle.blurple, row=1)
                    u8 = disnake.ui.Button(custom_id='u8', label='U8', style=disnake.ButtonStyle.blurple, row=1)
                    
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
                    
                    
                    u1.callback = u1callback
                    u2.callback = u2callback
                    u3.callback = u3callback
                    u4.callback = u4callback
                    u5.callback = u5callback
                    u6.callback = u6callback
                    u7.callback = u7callback
                    u8.callback = u8callback
                    
                    view = disnake.ui.View(timeout=None)
                    
                    view.add_item(u1)
                    view.add_item(u2)
                    view.add_item(u3)
                    view.add_item(u4)
                    view.add_item(u5)
                    view.add_item(u6)
                    view.add_item(u7)
                    view.add_item(u8)
                    
                    color = await average_color('imagecache/' + codeid + img_type)
                    
                    embed=disnake.Embed(title='Job Complete!', color=color)
                    embed.add_field(name='Content Filter:', value=str(vars['filter']), inline=True)
                    embed.add_field(name='Images timed out:', value=str(errors), inline=True)
                    embed.add_field(name='Filtered images:', value=str(filtered_images), inline=True)
                    embed.add_field(name='Code(for debugging):', value='||' + codeid + '||', inline=True)
                    embed.set_image(file=disnake.File('imagecache/' + codeid + img_type))
                    await message.delete()
                    await vars['inter_response'].reply('Job done for user ' + vars['userid'], embed=embed, view=view)
                else:
                    await message.edit(file=disnake.File('imagecache/' + codeid + img_type))

            elif use_embeds:
                embed=disnake.Embed(title='Job Failed', color=0xff0000)
                embed.add_field(name='Content Filter:', value=str(vars['nsfw']), inline=True)
                embed.add_field(name='Code(for debugging):', value='||' + codeid + '||', inline=True)
                await message.edit('Job done for ' + vars['userid'] + '!', embed=embed)
            else:
                await messaage.edit('Job Failed')
    await session.close()
    # Jesus christ thank god its finally over
    return        
    
bot.run(token)