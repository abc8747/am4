import asyncio
from pathlib import Path

import discord
from discord.ext import commands
from discord.ext.commands import Bot
from loguru import logger

import am4
from am4.utils import __version__ as am4utils_version
from am4.utils.db import init as utils_init

from ..config import cfg
from .channels import channels
from .cogs.aircraft import AircraftCog
from .cogs.airport import AirportCog
from .cogs.help import HelpCog
from .cogs.price import PriceCog
from .cogs.route import RouteCog
from .cogs.routes import RoutesCog
from .cogs.settings import SettingsCog
from .errors import CustomErrHandler

intents = discord.Intents.default()
intents.message_content = True
bot = Bot(command_prefix=cfg.bot.COMMAND_PREFIX, intents=intents, help_command=None)


@bot.event
async def on_ready():
    await channels.init(bot)
    logger.info(f"logged in as {bot.user} on {', '.join([g.name for g in bot.guilds])}")
    logger.info(f"am4.utils version {am4utils_version}")


@bot.event
async def on_guild_join(guild: discord.Guild):
    channel = guild.system_channel
    if channel is not None:
        embed = discord.Embed(
            title="Hello!",
            description=(
                "Here are some tips to setup the bot.\n\n"
                "1. Create two roles: `Easy` and `Realism` and assign them to users. "
                "The bot will pick them up automatically for accurate calculations.\n"
                "2. Join our [community server](https://discord.gg/4tVQHtf) and follow the "
                "`am4-bot-updates` channel. We will post updates on new features and bug fixes there.\n"
            ),
            colour=discord.Colour.blue(),
        )
        embed.add_field(
            name="Get started",
            value=(
                f"Show a list of all commands with `{cfg.bot.COMMAND_PREFIX}help`.\n"
                f"For example, try `{cfg.bot.COMMAND_PREFIX}help airport` or `{cfg.bot.COMMAND_PREFIX}help routes`.\n"
            ),
            inline=False,
        )
        embed.add_field(
            name="Contribute",
            value=(
                "[Source code](https://github.com/abc8747/am4), "
                "[list of formulae and documentation](https://abc8747.github.io/am4) "
                "is open-source on GitHub. Feel free to open an issue or pull request!"
            ),
            inline=False,
        )
        embed.set_thumbnail(url="https://am4help.com/assets/img/logo.png")
        await channel.send(embed=embed)


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    h = CustomErrHandler(ctx, error)
    await h.top_level()
    await h.raise_for_unhandled()


async def start(db_done: asyncio.Event):
    venv_path = am4.__path__[-1]
    if "site-packages" not in venv_path:
        logger.warning("am4 should be installed in a virtual environment")

    await db_done.wait()

    from .plots import MPLMap

    mpl_map = MPLMap()
    utils_init(home_dir=venv_path)
    await bot.add_cog(HelpCog(bot))
    await bot.add_cog(SettingsCog(bot))
    await bot.add_cog(AirportCog(bot))
    await bot.add_cog(AircraftCog(bot))
    await bot.add_cog(RouteCog(bot))
    await bot.add_cog(RoutesCog(bot, mpl_map))
    await bot.add_cog(PriceCog(bot))
    await bot.start(cfg.bot.DISCORD_TOKEN)
