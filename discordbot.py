#!/usr/bin/env python3

# requires the 'message_content' intent.

import os
import discord

import ai

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(msg):
    if msg.author == client.user:
        return

    q = msg.content
    if q.startswith('!') or 'visidata' in q.lower():
        await msg.channel.send(ai.main_query(q))


client.run(os.getenv('DISCORD_TOKEN'))
