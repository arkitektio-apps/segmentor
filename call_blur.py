from bergen.clients.default import Bergen
from bergen.models import Node
from bergen.enums import ClientType
import asyncio
from aiostream import stream, pipe, async_

async def main(loop):

    async with Bergen(
        host="arbeider",
        port=8000,
        client_id="OyTXRTXNLu6HQpegcch94eQScyrEC85tH0OkstKO", 
        client_secret="nhSPWLVe1Ub2UOEc231KL0KmCQkIpPGubcqJr176wYfSLgLshmJChPmAi7RPs7i1KifjyOmNrPild8VGvkUWfPkvy7dBWfgUPPo6QBTHSTjZluLngrCLg6NiVEF9hbgB",
        name="karl",
        loop=loop,
        client_type=ClientType.INTERNAL
    ):
        blur = await Node.asyncs.get(package="@canoncial/generic/filters", interface="gaussian-blur")

        x = await asyncio.gather(*[ blur({"rep": 1}) for i in range(0,1000)])
        print(x)





if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))
    loop.close()