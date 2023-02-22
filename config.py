servermodels = True
# If enabled, will fetch models from the server. If disabled, it will only
# use models specified in the model_list

model_list = ['stable_diffusion', 'Midjourney Diffusion', 'Furry Epoch', 'Yiffy', 'Zack3D', 'trinart', 'waifu_diffusion', 'Zeipher Female Model', 'Anything Diffusion', 'stable_diffusion_2.1', 'RPG', 'Poison', 'mo-di-diffusion', 'JWST Deep Space Diffusion', 'Borderlands', 'Van Gogh Diffusion', 'Ghibli Diffusion', 'Papercut Diffusion', 'Inkpunk Diffusion', 'Clazy', 'Classic Animation Diffusion', 'Comic-Diffusion', 'stable_diffusion_2.0', 'Arcane Diffusion', 'Spider-Verse Diffusion', 'Elden Ring Diffusion', 'Robo-Diffusion', 'Midjourney Diffusion', 'Redshift Diffusion', 'Mega Merge Diffusion', 'Cyberpunk Anime Diffusion', 'Samdoesarts Ultmerge', 'vectorartz', 'Knollingcase', 'Dungeons and Diffusion']
# List of models used, only applicable if servermodels is False,
# Use this if you want to limit usable models on the bot, but
# note that only stable horde models will work. Find this list at
# https://stablehorde.net/api/v2/status/models

token = 'MTA1NTk4NDY1NDc0Mzc3NzM5Mg.G0O528.4aQPpGc6dhf8pMmHSfj7xppaaFsviBspW6BUdo'
# discord bot token

sd_api_key = '0000000000'
# horde api key, planning to add /link, and /unlink

default_images = 4
# amount of images to generate on default, must be one of [1,2,4,6,8,9]


nsfw_filter = False
# if disabled, will disable the NSFW filter on the bot. Note: If disabled,
# it will only have the filter disabled in NSFW-marked channels

default_sampler = 'k_euler_a'
# default sampler for generations

default_model = 'stable_diffusion'
default_riff = 'pix2pix'
# default model for generations

default_width = 768
default_height = 768
# default width and height for generations, and when doing img2img,
# the default_width will be used as the shorter edge of the image to
# bring low resolution images up to a good img2img resolution

default_steps = 23
# default steps for generations

input_types = ('png', 'jpg', 'jpeg', 'webp', '.PNG', '.JPG', '.JPEG', '.WEBP')
img_type = '.webp'
format_type = 'WEBP'
# image file extension. Note: only PIL-supported types are supported right now.

use_embeds = True
# whether to use embeds to response to the user
# Idk why this is still here, pliss don't touch it
# I stopped actually making responses without embeds

accept_dm = True
# whether to accept requests in the DMs

save_nsfw = True
# if set to true, nsfw-filtered images will be saved in nsfwcache.

# ADVANCED SETTINGS:
    
url = 'https://stablehorde.net' # url to query
endpoint = '/api/v2/generate/' # api async endpoint
wait_time = 2 # time between asynchronous API calls, lower = more status information, 
              # higher = less bandwidth usage. Keep under 15 preferably

timeout = 580
# how long to wait before cancelling the request and sending images, keep below 600,
# but precious kudos will be sacrificed if you set it too low

showanywaytimeout = 100

error_color = '#ff0000'
nsfw_color = '#b449fb'
finished_color = '#58f000'
processing_color = '#f0e000'
restarted_color = '#00ecf0'
waiting_color = '#c300d1'
# Colors to use on the previews on images. Note: when changed, 
# remove all images in waitcache folder.

embed_icon = 'https://cdn.discordapp.com/avatars/963129050883325962/fc05c88bcc34d50fb6767372e6946132.png'
# Icon used on the initial embed!

filter_strength = 0.3
# Strength of nsfw filter. 0 will allow all messages, 1 will allow no messages

thinking_emoji = '🤔'
# Unicode Emoji

nsfw_neg_prompt = '(((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, fused animal ears, bad animal ears, poorly drawn animal ears, extra animal ears, liquidanimal ears, heavy animal ears, missing animal ears, text, ui, error, missing fingers, missing limb, fused fingers, one hand with more than 5 fingers, one hand with less than5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, colorful tongue, blacktongue, cropped, watermark, username, blurry, JPEG artifacts, signature, 3D, 3D game, 3D game scene, 3D character, malformed feet, extra feet, bad feet, poorly drawnfeet, fused feet, missing feet, extra shoes, bad shoes, fused shoes, more than two shoes, poorly drawn shoes, bad gloves, poorly drawn gloves, fused gloves, bad hairs, poorly drawn hairs, fused hairs, badeyes, fused eyes poorly drawn eyes, extra eyes, malformed limbs, more than 2 nipples, missing nipples, different nipples, fused nipples, bad nipples, poorly drawnnipples, black nipples, colorful nipples, gross proportions. short arm, (((missing arms))), missing thighs, missing calf, missing legs, mutation, duplicate, morbid, mutilated, poorly drawn hands, more than 1 left hand, more than 1 right hand, deformed, (blurry), disfigured, missing legs, extra arms, extra thighs, more than 2 thighs, extra calf,fused calf, extra legs, bad knee, extra knee, more than 2 legs, bad tails, bad mouth, fused mouth, poorly drawn mouth, bad tongue, tongue within mouth, too longtongue, black tongue, big mouth, cracked mouth, bad mouth, dirty face, dirty teeth, dirty pantie, fused pantie, poorly drawn pantie, fused cloth, poorly drawn cloth, badpantie, yellow teeth, thick lips, bad camel toe, colorful camel toe, bad asshole, poorly drawn asshole, fused asshole, missing asshole, bad anus, bad pussy, bad crotch, badcrotch seam, fused anus, fused pussy, fused anus, fused crotch, poorly drawn crotch, fused seam, poorly drawn anus, poorly drawn pussy, poorly drawn crotch, poorlydrawn crotch seam, bad thigh gap, missing thigh gap, fused thigh gap, liquid thigh gap, poorly drawn thigh gap, poorly drawn anus, bad collarbone, fused collarbone, missing collarbone, liquid collarbone, strong girl, obesity, worst quality, low quality, normal quality, liquid tentacles, bad tentacles, poorly drawn tentacles, split tentacles, fused tentacles, missing clit, bad clit, fused clit, colorful clit, black clit, liquid clit, QR code, bar code, censored, pubic hair, mosaic, futa, testis'

import json
import requests
# uses requests here, but asyncio in the bot.py, will fix later
import os

directories = ['textcache', 'upscalecache', 'nsfwcache', 'waitcache', 'imagecache']
# define directories

for path in directories:
    if not os.path.isdir(path):
        os.mkdir(path)
# loops through directories, if they don't exist, it makes them

print('Querying https://stablehorde.net/api/swagger.json This may take a minute.')
apidocs = requests.get('https://stablehorde.net/api/swagger.json').json()

sampler_list = apidocs['definitions']['ModelPayloadRootStable']['properties']['sampler_name']['enum']
processor_list = apidocs['definitions']['ModelPayloadRootStable']['properties']['post_processing']['items']['enum']

def get_top_models(data, num_models=25):
    # Sort the data by count value
    sorted_data = sorted(data, key=lambda x: x['count'], reverse=True)
    
    # Return the top num_models models
    return sorted_data[:num_models]

if servermodels:
    headers = {
    'accept': 'application/json',
    }
    print('Querying Servers... This may take a minute.')
    data = requests.get('https://stablehorde.net/api/v2/status/models', headers=headers)
    data = json.loads(data.text)
    filtered = 0
    for i in range(len(data)):
        i = i-filtered
        
        model = data[i]['name']
        if model == 'stable_diffusion_inpainting' or model == 'Stable Diffusion 2 Depth':
            data.pop(i)
            print('Warning: Model is an inpainting/depth2img model. It will not be added \n')
            filtered += 1

    sorted_data = get_top_models(data)
    model_list = []
    for i in range(len(sorted_data)):
        model = sorted_data[i]['name']
        print('Model found: ' + model)
        model_list.append(model)
            
    print('Model List: ' + str(model_list))
else:
    print('\n Using User-defined list: ' + str(model_list) + '\n')

if len(model_list) > 25:
    print('Error: List is greater than max allowed by discord. Using top 25 models')
    model_list = model_list[:25]