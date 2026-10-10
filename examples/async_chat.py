"""Async library usage."""

import asyncio

from freewhirr import Freewhirr


async def main() -> None:
    async with Freewhirr() as client:
        result = await client.acomplete(
            messages=[{"role": "user", "content": "Name three flightless birds."}]
        )
        print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
