import os.path as osp, glob, numpy as np, sys, os, glob
import tqdm
import json
import uptools
import seutils
from math import pi
import awkward as ak
import bbefp

np.random.seed(1001)


def ntup_to_npz_signal(event, outfile):
    select_zjet = event[b'ak15GenJetsPackedNoNu_energyFromZ'].argmax()
    zjet = uptools.Vectors.from_prefix(b'ak15GenJetsPackedNoNu', event, branches=[b'energyFromZ'])[select_zjet]
    if zjet.energyFromZ / zjet.energy < 0.01:
        # print('Skipping event: zjet.energyFromZ / zjet.energy = ', zjet.energyFromZ / zjet.energy)
        return False
    constituents = (
        uptools.Vectors.from_prefix(b'ak15GenJetsPackedNoNu_constituents', event, b'isfromz')
        .flatten()
        .unflatten(event[b'ak15GenJetsPackedNoNu_nConstituents'])
        )[select_zjet]
    constituents = constituents.flatten()
    if not osp.isdir(osp.dirname(outfile)): os.makedirs(osp.dirname(outfile))
    np.savez(
        outfile,
        pt = constituents.pt,
        eta = constituents.eta,
        phi = constituents.phi,
        energy = constituents.energy,
        y = 1,
        jet_pt = zjet.pt,
        jet_eta = zjet.eta,
        jet_phi = zjet.phi,
        jet_energy = zjet.energy,        
        )
    return True

def ntup_to_npz_bkg(event, outfile):
    '''
    Just dumps the two leading jets to outfiles
    '''
    jets = uptools.Vectors.from_prefix(b'ak15GenJetsPackedNoNu', event)
    constituents = (
        uptools.Vectors.from_prefix(b'ak15GenJetsPackedNoNu_constituents', event)
        .flatten()
        .unflatten(event[b'ak15GenJetsPackedNoNu_nConstituents'])
        )
    for i in [0, 1]:
        # Do leading and subleading
        this_jet = jets[i]
        this_constituents = constituents[i].flatten()
        title = ['leading', 'subleading'][i]
        if not osp.isdir(osp.dirname(outfile)): os.makedirs(osp.dirname(outfile))
        np.savez(
            outfile.replace('.npz', '_'+ title + '.npz'),
            pt = this_constituents.pt,
            eta = this_constituents.eta,
            phi = this_constituents.phi,
            energy = this_constituents.energy,
            y = 0,
            jet_pt = this_jet.pt,
            jet_eta = this_jet.eta,
            jet_phi = this_jet.phi,
            jet_energy = this_jet.energy,
            )

def iter_arrays_qcd(N):
    xss = [0.23*7025.0, 620.6, 59.06, 18.21, 7.5, 0.6479, 0.08715, 0.005242, 0.0001349, 3.276]
    ntuple_path = 'root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/boosted/ecfntuples/'
    samples = [
        'Feb15_qcd_BTV-RunIIFall18GS-00024_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00025_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00026_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00027_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00029_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00030_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00031_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00032_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00033_1_cfg',
        'Feb15_qcd_BTV-RunIIFall18GS-00051_1_cfg',
        ]
    rootfiles = [ seutils.ls_wildcard(ntuple_path+s+'/*.root') for s in samples ]
    yield from uptools.iter_arrays_weighted(N, xss, rootfiles, treepath='gensubntupler/tree')

def iter_events_qcd(N):
    for arrays in iter_arrays_qcd(N):
        for i in range(uptools.numentries(arrays)):
            yield uptools.get_event(arrays, i)

def make_npzs_bkg(N=12000):
    train_outdir = 'data/train/raw/qcd'
    test_outdir = 'data/test/raw/qcd'
    for i_event, event in tqdm.tqdm(enumerate(iter_events_qcd(N)), total=N):
        outdir = test_outdir if i_event % 5 == 0 else train_outdir
        ntup_to_npz_bkg(event, outdir + f'/{i_event}.npz')
    print(f'Bkg npzs done - {N} events')


def make_npzs_signal(directory):
    N = seutils.root.sum_dataset(directory, treepath='gensubntupler/tree')
    signal = seutils.ls_wildcard(directory + '/*.root')
    signal_name = osp.basename(osp.dirname(signal[0]))
    train_outdir = 'data/train/raw/' + signal_name
    test_outdir = 'data/test/raw/' + signal_name
    n_total = 0
    i_event_good = 0
    for i_event, event in tqdm.tqdm(
        enumerate(uptools.iter_events(signal, treepath='gensubntupler/tree', nmax=N)),
        total=N
        ):
        outdir = test_outdir if i_event_good % 5 == 0 else train_outdir
        succeeded = ntup_to_npz_signal(event, outdir + f'/{i_event_good}.npz')
        if succeeded: i_event_good += 1
        n_total += 1
    print(
        'Total events turned into npzs: {}/{}  ({} failures due to 0 Z-energy)'
        .format(i_event_good, n_total, n_total-i_event_good)
        )

def _iter_npzs(rawdir, nmax=None):
    """Iterate over the npz files, returning nested ordinary python lists"""
    npzs = glob.glob(rawdir + '/*/*.npz')
    N = len(npzs) if nmax is None else min(nmax, len(npzs))
    for i, npz in tqdm.tqdm(enumerate(sorted(npzs)), total=N):
        if nmax and i==nmax: break
        d = np.load(npz)
        get = lambda key: d[key].tolist()
        yield [
            get('y'),
            get('pt'), get('eta'), get('phi'), get('energy'),
            get('jet_pt'), get('jet_eta'), get('jet_phi'), get('jet_energy')
            ]

def ak_transpose(array):
    """
    Transposes the first two dimensions of an awkward array.
    Useful for (n_events x n_features) --> (n_features x n_events)
    or  (n_features x n_events) --> (n_events x n_features)
    """
    return ak.fromiter(array[:,i] for i in range(len(array[0])))

def merge_npzs_to_ak(rawdir, outfile=None, nmax=None):
    """
    Loops over all .npz files in rawdir, stacks all events into an ak array,
    and dumps it to a file.
    """
    if outfile is None: outfile = osp.dirname(rawdir) + '/merged.awkd'
    bbefp.logger.info(f'Merging {rawdir} --> {outfile}')
    merged = ak.fromiter(_iter_npzs(rawdir, nmax))
    ak.save(outfile, ak_transpose(merged))


# __________________________________________________
class Bunch:
    def __init__(self, **kwargs):
        self.arrays = kwargs

    def __getattr__(self, name):
       return self.arrays[name]

    def __getitem__(self, where):
        """Selection mechanism"""
        new = self.__class__()
        new.arrays = {k: v[where] for k, v in self.arrays.items()}
        return new

    def __len__(self):
        for k, v in self.arrays.items():
            try:
                return len(v)
            except TypeError:
                return 1


class FourVectorArray:
    """
    Wrapper class for Bunch, with more specific 4-vector stuff
    """
    def __init__(self, pt, eta, phi, energy, **kwargs):
        self.bunch = Bunch(
            pt=pt, eta=eta, phi=phi, energy=energy, **kwargs
            )

    def __getattr__(self, name):
       return getattr(self.bunch, name)

    def __getitem__(self, where):
        new = self.__class__([], [], [], [])
        new.bunch = self.bunch[where]
        return new

    def __len__(self):
        return len(self.bunch)

    @property
    def rapidity(self):
        return rapidity(self.pt, self.eta, self.energy)


def rapidity(pt, eta, energy):
    pz = pt * np.sinh(eta) 
    return 0.5 * np.log((energy + pz) / (energy - pz))


def is_array(a):
    """
    Checks if a thing is an array or maybe a number
    """
    try:
        shape = a.shape
        return len(shape) >= 1
    except AttributeError:
        return False


def calc_dphi(phi1, phi2):
    """
    Calculates delta phi. Assures output is within -pi .. pi.
    """
    twopi = 2.*np.pi
    # Map to 0..2pi range
    dphi = (phi1 - phi2) % twopi
    # Map pi..2pi --> -pi..0
    if is_array(dphi):
        select = (dphi > np.pi)
        dphi[select] = dphi[select] - twopi
    elif dphi > np.pi:
        dphi -= twopi
    return dphi


def calc_dr(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1-eta2)**2 + calc_dphi(phi1, phi2)**2)



# __________________________________________________
import torch
from torch_geometric.data import (Data, Dataset, DataLoader)

class DRNDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""

    def __init__(self, array, *args, **kwargs):
        if isinstance(array, str): array = ak.load(array)
        super(DRNDataset, self).__init__('whatever', *args, **kwargs)
        self.y = array[0]
        self.constituents = FourVectorArray(array[1], array[2], array[3], array[4])
        self.jets = FourVectorArray(array[5], array[6], array[7], array[8])
        # Overwrite with delta's w.r.t. to the main jet
        self.constituents.phi = calc_dphi(self.constituents.phi, self.jets.phi)
        self.constituents.eta = self.constituents.eta - self.jets.eta

    def __len__(self):
        return len(self.y)

    def get(self, i):
        X = np.stack((
            self.constituents[i].pt,
            self.constituents[i].eta,
            self.constituents[i].phi,
            self.constituents[i].energy
            )).T
        return Data(
            x = torch.from_numpy(X),
            y = torch.from_numpy(self.y[i:i+1])
            )

# __________________________________________________
class ParticleNetDataset(torch.utils.data.Dataset):
    def __init__(self, array, n_constituents=200):
        if isinstance(array, str): array = ak.load(array)
        self.n_constituents = n_constituents
        self.y = array[0]
        self.constituents = FourVectorArray(array[1], array[2], array[3], array[4])
        self.jets = FourVectorArray(array[5], array[6], array[7], array[8])
        # Overwrite with delta's w.r.t. to the main jet
        self.constituents.phi = calc_dphi(self.constituents.phi, self.jets.phi)
        self.constituents.eta = self.constituents.eta - self.jets.eta

        # features
        self.constituents.logpt = np.log(self.constituents.pt)
        self.constituents.loge = np.log(self.constituents.energy)
        self.constituents.logpt_ptjet = np.log(self.constituents.pt/self.jets.pt)
        self.constituents.loge_ejet = np.log(self.constituents.energy/self.jets.energy)
        self.constituents.dr = np.sqrt(self.constituents.eta**2 + self.constituents.phi**2)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        features = np.zeros((5, self.n_constituents))
        points = np.zeros((2, self.n_constituents))
        mask = np.zeros((1, self.n_constituents))

        constituents = self.constituents[i]
        n_constituents_this = min(len(constituents), self.n_constituents)

        features_nonpad = np.stack((
            self.constituents.logpt[i],
            self.constituents.loge[i],
            self.constituents.logpt_ptjet[i],
            self.constituents.loge_ejet[i],
            self.constituents.dr[i]
            ))
        assert features_nonpad.shape[0] == 5
        features[:,:n_constituents_this] = features_nonpad[:,:n_constituents_this]
        assert features.shape == (5, self.n_constituents)

        points[:,:n_constituents_this] = np.stack((self.constituents.phi[i], self.constituents.eta[i]))[:,:n_constituents_this]
        mask[0, :n_constituents_this] = 1

        return (
            torch.from_numpy(points).type(torch.FloatTensor),
            torch.from_numpy(features).type(torch.FloatTensor),
            torch.from_numpy(mask).type(torch.FloatTensor),
            torch.from_numpy(self.y[i:i+1]).type(torch.LongTensor)
            )



# def get_loader_ptm(merged_npz, batch_size):
#     """For torch datasets"""
#     import torch
#     d = np.load(merged_npz)
#     dataset = torch.utils.data.TensorDataset(
#         torch.from_numpy(d['X_ptm']),
#         torch.from_numpy(d['y'])
#         )
#     return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# def get_loader_ptm_plus_efps(merged_npz, efp_file, efps, batch_size):
#     import torch
#     d = np.load(merged_npz)
#     ptm = d['X_ptm']
#     efp_values = np.load(efp_file)['efp'][:,efps]
#     efp_values /= np.max(efp_values)
#     X = np.hstack((ptm, efp_values)).astype(np.float32)
#     dataset = torch.utils.data.TensorDataset(
#         torch.from_numpy(X),
#         torch.from_numpy(d['y'])
#         )
#     return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def cli():
    import argparse, shutil
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action', type=str,
        choices=['npzs', 'merge'],
        )
    parser.add_argument(
        'which_to_do', type=str,
        choices=['sig', 'bkg', 'both'],
        nargs='?', default='both'
        )
    args = parser.parse_args()

    do_sig = args.which_to_do in [ 'sig', 'both' ]
    do_bkg = args.which_to_do in [ 'bkg', 'both' ]

    def rmdir(directory):
        if osp.isdir(directory): shutil.rmtree(directory)

    def rm(file):
        if osp.isfile(file): os.remove(file)


    if args.action == 'npzs':
        if do_sig:
            rmdir('data/test/raw/Mar16_mz150_rinv0.1_mdark10')
            rmdir('data/train/raw/Mar16_mz150_rinv0.1_mdark10')
            make_npzs_signal('root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/boosted/ecfntuples/Mar16_mz150_rinv0.1_mdark10')
        if do_bkg:
            rmdir('data/test/raw/qcd')
            rmdir('data/train/raw/qcd')
            make_npzs_bkg()

    elif args.action == 'merge':
        rm('data/train/merged.awkd')
        rm('data/test/merged.awkd')
        merge_npzs_to_ak('data/train/raw')
        merge_npzs_to_ak('data/test/raw')


if __name__ == '__main__':
    cli()