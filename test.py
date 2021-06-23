from bergen.clients.host import HostBergen
from bergen.models import Pod
import logging
from grunnlag.schema import Representation
import sys
import os
import namegenerator

logger = logging.getLogger(__name__)



def main():

    unique_name = os.getenv("ARNHEIM_NAME", namegenerator.gen())
    pod_id = int(os.getenv("ARNHEIM_POD_ID"))

    client = HostBergen(
            port=8000,
            name=unique_name,# if we want to specifically only use pods on this innstance we would use that it in the selector
    )


    
    segmentor = Pod.objects.get(id=int(pod_id))


    @client.register(segmentor, gpu=True)
    def segmentor(helper, rep: Representation = None, slice = None):
        """Sleep on the CPU

        Args:
            helper ([type]): [description]
            rep ([type], optional): [description]. Defaults to None.
        """
        print("Called")
        nana = Representation.objects.from_xarray(rep.data.max(dim="z"), name="segmented", sample=rep.sample.id)
        return {"rep": nana}


    client.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down everything Relentlessly")
        sys.exit(1)

