import discord
from discord import AllowedMentions, Embed
from discord.ext import commands

from am4.utils import __version__ as am4utils_version

from ...config import cfg
from ..base import BaseCog
from ..utils import COLOUR_ERROR, COLOUR_GENERIC


class HelpCog(BaseCog):
    @commands.command(brief="Shows information about our bot", ignore_extra=False)
    async def botinfo(self, ctx: commands.Context):
        embed = Embed(title=f"AM4Help `v{am4utils_version}`", color=discord.Colour.blue())
        embed.add_field(name="Made by", value="<@697804580456759397>", inline=False)
        embed.add_field(
            name="Contributed by",
            value="<@243007616714801157>, <@663796476257763370>, Star Alliance members and the community",
            inline=False,
        )
        embed.add_field(
            name="Support",
            value="If you need any help, join our [community server](https://discord.gg/4tVQHtf)!",
            inline=False,
        )
        embed.add_field(
            name="Contribute",
            value=(
                "[Source code](https://github.com/cathaypacific8747/am4), "
                "[list of formulae and documentation](https://cathaypacific8747.github.io/am4) "
                "is open-source on GitHub. Feel free to open an issue or pull request!"
            ),
            inline=False,
        )
        embed.set_thumbnail(url="https://am4help.com/assets/img/logo.png")
        await ctx.send(embed=embed, allowed_mentions=AllowedMentions.none())

    @staticmethod
    def _collect_commands_descs(cmd: commands.Command | commands.Group, level=0):
        if not cmd.hidden and cmd.enabled:
            cmd_display = f"{cfg.bot.COMMAND_PREFIX if level == 0 else ''}{'  ' * level}{cmd.name}"
            desc = f"`{cmd_display:<10}`{'  ' * level} - {cmd.short_doc}\n"
            if isinstance(cmd, commands.Group):
                desc += "".join(HelpCog._collect_commands_descs(subcmd, level + 1) for subcmd in cmd.commands)
            return desc
        return ""

    @commands.command(brief="Shows this command", ignore_extra=False)
    async def help(
        self,
        ctx: commands.Context,
        command: str | None = commands.parameter(default=None, description="Name of the command to show help for"),
        *subcommands: str,
    ):
        if command is None:
            embed = discord.Embed(title="List of commands", colour=COLOUR_GENERIC)
            desc = "".join(HelpCog._collect_commands_descs(cmd) for cmd in self.bot.commands)
            embed.description = desc
            embed.add_field(
                name="Learn more about how to use a command with `$help <cmd>`.",
                value="For example, try:```php\n$help airport\n$help settings show```",
            )
            await ctx.send(embed=embed)
            return

        cmd = self.bot.get_command(command)
        if cmd is None:
            await ctx.send(
                embed=discord.Embed(
                    title=f"Command `{command}` does not exist",
                    description="To show all commands, use `$help`.",
                    colour=COLOUR_ERROR,
                )
            )
            return
        for sc in subcommands:
            cmd = cmd.get_command(sc)
            if cmd is None:
                await ctx.send(
                    embed=discord.Embed(
                        title=f"Subcommand `{sc}` does not exist!",
                        description=f"To show all possible subcommands, use `$help {command}`.",
                        colour=COLOUR_ERROR,
                    )
                )
                return

        embed = discord.Embed(
            title=f"`{cfg.bot.COMMAND_PREFIX}{cmd.qualified_name} {cmd.signature}`", colour=COLOUR_GENERIC
        )
        for p in cmd.clean_params.values():
            name = ":small_blue_diamond: `{}` (Optional)" if not p.required else ":small_orange_diamond: `{}`"
            embed.add_field(
                name=name.format(p.name),
                value=d if (d := p.description) is not None else "",
                inline=True,
            )
        if h := cmd.help or cmd.brief:
            embed.add_field(name="Details", value=h, inline=False)
        await ctx.send(embed=embed)
