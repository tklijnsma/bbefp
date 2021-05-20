import numpy as np


def np_to_structuredarray(event):
    event = event[~np.all((event == 0.), axis=1)]
    return np.array(
        [tuple(e) for e in event],
        dtype=np.dtype([('E', 'f8'), ('px', 'f8'), ('py', 'f8'), ('pz', 'f8')])
        )


def _cluster(epxpypyz, return_np=True, **kwargs):
    import pyjet
    kwargs.setdefault('algo', 'antikt')
    kwargs.setdefault('R', .3)
    kwargs.setdefault('ep', True)
    jets = pyjet.cluster(
        np_to_structuredarray(epxpypyz),
        algo='antikt', R=0.3, ep=True
        ).inclusive_jets()
    if return_np:
        return np.array([[j.e, j.px, j.py, j.pz] for j in jets])
    else:
        return jets


def cluster(events, append_zeros=True, **kwargs):
    kwargs.setdefault('return_np', True)
    return_zeroth = False
    if len(events.shape) == 2:
        events = [events]
        return_zeroth = True
    returnable = [ _cluster(event, **kwargs) for event in events ]
    if kwargs['return_np']:
        if return_zeroth: return np.array(returnable[0])
        if not append_zeros: return np.array(returnable)
        maxdim = max(map(len, returnable))
        out = np.zeros((events.shape[0], maxdim, 4))
        for i, m in enumerate(returnable):
            out[i,:m.shape[0]] = m
        return out
    else:
        if return_zeroth: return returnable[0]
        return returnable


