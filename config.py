servermodels = True
# If enabled, will fetch models from the server. If disabled, it will only
# use models specified in the model_list

model_list = ['stable_diffusion', 'Midjourney Diffusion', 'Furry Epoch', 'Yiffy', 'Zack3D', 'trinart', 'waifu_diffusion', 'Zeipher Female Model', 'Anything Diffusion', 'stable_diffusion_2.1', 'RPG', 'Poison', 'mo-di-diffusion', 'JWST Deep Space Diffusion', 'Borderlands', 'Van Gogh Diffusion', 'Ghibli Diffusion', 'Papercut Diffusion', 'Inkpunk Diffusion', 'Clazy', 'Classic Animation Diffusion', 'Comic-Diffusion', 'stable_diffusion_2.0', 'Arcane Diffusion', 'Spider-Verse Diffusion', 'Elden Ring Diffusion', 'Robo-Diffusion', 'Midjourney Diffusion', 'Redshift Diffusion', 'Mega Merge Diffusion', 'Cyberpunk Anime Diffusion', 'Samdoesarts Ultmerge', 'vectorartz', 'Knollingcase', 'Dungeons and Diffusion']
# List of models used, only applicable if servermodels is False,
# Use this if you want to limit usable models on the bot, but
# note that only stable horde models will work. Find this list at
# https://stablehorde.net/api/v2/status/models

token = 'BOT_TOKEN_HERE'
# discord bot token

sd_api_key = '0000000000'
# default horde api key, optional but recommended

nsfw_filter = False
# if disabled, will disable the NSFW filter on the bot. Note: If disabled,
# it will only have the filter disabled in NSFW-marked channels

default_sampler = 'k_euler_a'
# default sampler for generations

default_model = 'stable_diffusion'
# default model for generations

default_width = 576
default_height = 576
# default width and height for generations, and when doing img2img,
# the default_width will be used as the shorter edge of the image to
# bring low resolution images up to a good img2img resolution

default_steps = 15
# default steps for generations

input_types = ('.png', '.jpg', '.jpeg', '.webp', '.PNG', '.JPG', '.JPEG', '.WEBP')
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

filter_strength = 0.4
# Strength of nsfw filter. 0 will allow all messages, 1 will allow no messages

import json
import requests

print('Querying https://stablehorde.net/api/swagger.json This may take a minute.')
apidocs = requests.get('https://stablehorde.net/api/swagger.json').json()

sampler_list = apidocs['definitions']['ModelPayloadRootStable']['properties']['sampler_name']['enum']
processor_list = apidocs['definitions']['ModelPayloadRootStable']['properties']['post_processing']['items']['enum']

if servermodels:
    headers = {
    'accept': 'application/json',
    }
    print('Querying Servers... This may take a minute.')
    data = requests.get('https://stablehorde.net/api/v2/status/models', headers=headers)
    data = json.loads(data.text)
    model_list = []
    for i in range(len(data)):
        model = data[i]['name']
        print('Model found: ' + model)
        if not model == 'stable_diffusion_inpainting':
            model_list.append(model)
        else:
            print('The above model is an inpainting model. It will not be added \n')
            
    print('Model List: ' + str(model_list))
else:
    print('\n Using User-defined list: ' + str(model_list) + '\n')

if len(model_list) > 24:
    print('Error: List is greater than max allowed by discord. Using top 25 models')
    model_list = model_list[:24]
