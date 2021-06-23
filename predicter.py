
from stardist.models import StarDist3D
from csbdeep.utils import Path, normalize
import sys
import numpy as np
import xarray as xr

from grunnlag.schema import Representation, RepresentationVariety
from bergen.clients.provider import ProviderBergen
from bergen import use



model = StarDist3D(None, name='stardist2', basedir='models')

def main():

    with ProviderBergen(
        config_path="bergen.yaml",
        log_stream= True,
        force_new_token=True,
        auto_reconnect=True# if we want to specifically only use pods on this innstance we would use that it in the selector
    ) as client:


        @client.provider.enable(gpu=True)
        async def segmentor(rep: Representation) -> Representation:
            """Segmentor

            Segments Cells

            Args:
                rep (Representation): The Representation. 

            Returns:
                Representation: A Representation
            
            """
            print("Called wtih Rep")
            
            axis_norm = (0,1,2) 

            x = rep.data.sel(c=0, t=0).transpose(*"zxy").data.compute()
            x = normalize(x,1,99.8,axis=axis_norm)

            labels, details = model.predict_instances(x)
            
            array = xr.DataArray(labels, dims=list("zxy")).expand_dims("c").expand_dims("t").transpose(*"xyzct")

            nana = await Representation.asyncs.from_xarray(array, name="Segmented " + rep.name, sample=rep.sample, variety=RepresentationVariety.MASK)
            return nana


        client.provide()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)





