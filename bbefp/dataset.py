import os.path as osp, glob, numpy as np, sys, os, glob
import tqdm
import json
import uptools
import seutils
from math import pi

np.random.seed(1001)


class ExtremaRecorder():
    '''
    Records the min and max of a set of values
    Doesn't store the values unless the standard deviation is requested too
    '''
    def __init__(self, do_std=True):
        self.min = 1e6
        self.max = -1e6
        self.mean = 0.
        self.n = 0
        self.do_std = do_std
        if self.do_std: self.values = np.array([])

    def update(self, values):
        self.min = min(self.min, np.min(values))
        self.max = max(self.max, np.max(values))
        n_new = self.n + len(values)
        self.mean = (self.mean * self.n + np.sum(values)) / n_new
        self.n = n_new
        if self.do_std: self.values = np.concatenate((self.values, values))

    def std(self):
        if self.do_std:
            return np.std(self.values)
        else:
            return 0.

    def __repr__(self):
        return (
            '{self.min:+7.3f} to {self.max:+7.3f}, mean={self.mean:+7.3f}{std} ({self.n})'
            .format(
                self=self,
                std='+-{:.3f}'.format(self.std()) if self.do_std else ''
                )
            )

    def hist(self, outfile):
        import matplotlib.pyplot as plt
        figure = plt.figure()
        ax = figure.gca()
        ax.hist(self.values, bins=25)
        plt.savefig(outfile, bbox_inches='tight')


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

def make_npzs_bkg(N=8000):
    train_outdir = 'data/train/raw/qcd'
    test_outdir = 'data/test/raw/qcd'
    for i_event, event in tqdm.tqdm(enumerate(iter_events_qcd(N)), total=N):
        outdir = test_outdir if i_event % 5 == 0 else train_outdir
        ntup_to_npz_bkg(event, outdir + f'/{i_event}.npz')


def make_npzs_signal():
    directory = 'root://cmseos.fnal.gov//store/user/lpcsusyhad/SVJ2017/boosted/ecfntuples/Mar16_mz150_rinv0.1_mdark10'
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


def rapidity(pt, eta, energy):
    pz = pt * np.sinh(eta) 
    return 0.5 * np.log((energy + pz) / (energy - pz))


def npz_to_features(d):
    pt = d['pt']
    z = pt / np.sum(pt)

    # deta = d['eta'] - d['jet_eta']
    # deta /= 2.

    jet_rapidity = rapidity(d['jet_pt'], d['jet_eta'], d['jet_energy'])
    crapidity = rapidity(pt, d['eta'], d['energy'])
    drapidity = crapidity - jet_rapidity

    dphi = d['phi'] - d['jet_phi']
    dphi %= 2.*pi # Map to 0..2pi range
    dphi[dphi > pi] = dphi[dphi > pi] - 2.*pi # Map >pi to -pi..0 range
    # dphi /= 2. # Normalize to approximately -1..1 range
    #            # (jet size is 1.5, but some things extend up to 2.)

    return z, drapidity, dphi


def npz_to_epxpypx(d):
    A = uptools.FourVectorArray(d['pt'], d['eta'], d['phi'], d['energy'])
    return np.stack((A.energy, A.px, A.py, A.pz)).T


def extrema(rawdir):
    ext_z = ExtremaRecorder()
    ext_drapidity = ExtremaRecorder()
    ext_dphi = ExtremaRecorder()
    rawfiles = glob.glob(rawdir + '/*/*.npz')
    np.random.shuffle(rawfiles)
    for npz in tqdm.tqdm(rawfiles[:3000], total=3000):
        d = np.load(npz)
        z, drapidity, dphi = npz_to_features(d)
        ext_z.update(z)
        ext_drapidity.update(drapidity)
        ext_dphi.update(dphi)
    print('Max dims:')
    print('z:         ', ext_z)
    print('drapidity: ', ext_drapidity)
    print('dphi:      ', ext_dphi)
    ext_z.hist('extrema_z.png')
    ext_drapidity.hist('extrema_drapidity.png')
    ext_dphi.hist('extrema_dphi.png')


def merge_npzs(rawdir):
    outfile = osp.dirname(rawdir) + '/merged.npz'
    max_dim = 0
    npzs = glob.glob(rawdir + '/*/*.npz')

    merged_y = []
    merged_inpz = []
    merged_jet4vecs = []
    proto_X = []
    proto_epxpypz = []
    ptetaphie = []
    for i_npz, npz in tqdm.tqdm(enumerate(sorted(npzs)), total=len(npzs)):
        d = np.load(npz)
        merged_y.append(d['y'])
        merged_inpz.append(i_npz)
        merged_jet4vecs.append([d['jet_pt'], d['jet_eta'], d['jet_phi'], d['jet_energy']])
        z, drapidity, dphi = npz_to_features(d)
        max_dim = max(z.shape[0], max_dim)
        proto_X.append([z, drapidity, dphi])
        ptetaphie.append(np.stack((d['pt'], d['eta'], d['phi'], d['energy'])).T)
        proto_epxpypz.append(npz_to_epxpypx(d))

    # Since X size is only known after full loop, only construct it now
    merged_X = []
    for z, drapidity, dphi in proto_X:
        X = np.zeros((max_dim, 3))
        n_this_jet = z.shape[0]
        X[:n_this_jet,0] = z
        X[:n_this_jet,1] = drapidity
        X[:n_this_jet,2] = dphi
        merged_X.append(X)

    # Same for epxpypz: Need to add zeroes at the end
    merged_epxpypz = []
    for epxpypz in proto_epxpypz:
        X = np.zeros((max_dim, 4))
        n_this_jet = epxpypz.shape[0]
        X[:n_this_jet] = epxpypz
        merged_epxpypz.append(X)

    # List of jet pt and mass for the EFP finding algo
    merged_jet4vecs = np.array(merged_jet4vecs)
    jets = uptools.FourVectorArray(*(merged_jet4vecs.T))
    merged_X_ptm = np.stack((jets.pt, jets.mass)).T

    merged_X = np.array(merged_X)
    merged_epxpypz = np.array(merged_epxpypz)
    merged_y = np.array(merged_y)
    merged_inpz = np.array(merged_inpz)

    # Shuffle
    random_order = np.random.permutation(merged_X.shape[0])
    merged_X = merged_X[random_order]
    merged_X_ptm = merged_X_ptm[random_order]
    merged_y = merged_y[random_order]
    merged_inpz = merged_inpz[random_order]

    np.savez(
        outfile,
        X=merged_X, X_ptm=merged_X_ptm, epxpypz=merged_epxpypz,
        y=merged_y, inpz=merged_inpz, ptetaphie=ptetaphie, jet4vec=merged_jet4vecs
        )


def get_loader_ptm(merged_npz, batch_size):
    """For torch datasets"""
    import torch
    d = np.load(merged_npz)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(d['X_ptm']),
        torch.from_numpy(d['y'])
        )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

def get_loader_ptm_plus_efps(merged_npz, efp_file, efps, batch_size):
    import torch
    d = np.load(merged_npz)
    ptm = d['X_ptm']
    efp_values = np.load(efp_file)['efp'][:,efps]
    efp_values /= np.max(efp_values)
    X = np.hstack((ptm, efp_values)).astype(np.float32)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(d['y'])
        )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


import torch
from torch_geometric.data import (Data, Dataset, DataLoader)

class ZNNDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""

    def __init__(self, merged_npz, *args, **kwargs):
        d = np.load(merged_npz, allow_pickle=True)
        self.ptetaphie = d['ptetaphie']
        self.jet4vecs = d['jet4vec']
        self.y = d['y']
        super(ZNNDataset, self).__init__(merged_npz, *args, **kwargs)
        # self.__indices__ = range(len(self.ptetaphie))
        # self.transform = None

    def norm(self, X, jet):
        pt, eta, phi, e = X.T
        deta_norm = (eta - jet[1])/2.

        dphi = phi - jet[2]
        dphi %= 2.*pi # Map to 0..2pi range
        dphi[dphi > pi] = dphi[dphi > pi] - 2.*pi # Map >pi to -pi..0 range
        dphi_norm = dphi/2. # Normalize to approximately -1..1 range

        pt_norm = pt / 30.
        energy_norm = e / 100.
        return np.stack((pt_norm, deta_norm, dphi_norm, energy_norm)).T


    def __len__(self):
        return len(self.ptetaphie)

    @property
    def raw_file_names(self):
        return [self.merged_npz]    
    
    def get(self, i):
        return Data(
            x = torch.from_numpy(self.norm(self.ptetaphie[i], self.jet4vecs[i])),
            y = torch.from_numpy(self.y[i:i+1])
            )

def get_loader_ptetaphie(merged_npz, batch_size):
    dataset = ZNNDataset(merged_npz)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)




def calc_dphi(a, b):
    dphi = a - b
    dphi %= 2.*pi # Map to 0..2pi range
    try:
        dphi[dphi > pi] = dphi[dphi > pi] - 2.*pi # Map >pi to -pi..0 range
    except TypeError:
        if dphi > pi: dphi -= 2.*pi # In case a and b were both simple floats
    return dphi


def pad_zeros(a, N):
    """
    Adds zeros or cuts away parts of a

    a: dim-1 array to be padded
    N: Dim to be padded to, or to be cut to
    """
    return a[:N] if a.shape[0] > N else np.concatenate((a, np.zeros(N-a.shape[0])))

def get_data_particlenet(merged_npz, N=200):
    d = np.load(merged_npz, allow_pickle=True)
    ptetaphie = d['ptetaphie']
    jet4vecs = d['jet4vec']

    n_events = len(jet4vecs)

    # Get the target dimension if not given
    if N is None: N = max(e.shape[0] for e in ptetaphie)
   
    all_coords = np.zeros((n_events, N, 2))
    all_features = np.zeros((n_events, N, 5))
    all_masks = np.zeros((n_events, N, 1))

    for i in range(n_events):
        pt, eta, phi, e = ptetaphie[i].T
        jet_pt, jet_eta, jet_phi, jet_e = jet4vecs[i]

        # coords
        deta = eta - jet_eta
        dphi = calc_dphi(phi, jet_phi)

        # features
        logpt = np.log(pt)
        loge = np.log(e)
        logpt_ptjet = np.log(pt/jet_pt)
        loge_ejet = np.log(e/jet_e)
        dr = np.sqrt(deta**2 + dphi**2)

        all_features[i] = np.stack((
            pad_zeros(logpt, N),
            pad_zeros(loge, N),
            pad_zeros(logpt_ptjet, N),
            pad_zeros(loge_ejet, N),
            pad_zeros(dr, N),
            )).T

        all_coords[i] = np.stack((
            pad_zeros(deta, N),
            pad_zeros(dphi, N)
            )).T

        all_masks[i,:len(pt),:] = 1

    y = np.stack((1-d['y'], d['y'])).T # One-hot encoded
    return dict(points=all_coords, features=all_features, mask=all_masks), y


def main():
    import argparse, shutil
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action', type=str,
        choices=['extrema', 'makenpzs', 'mergenpzs', 'fromscratch'],
        )
    args = parser.parse_args()

    if args.action == 'makenpzs' or args.action == 'fromscratch':
        if osp.isdir('data'): shutil.rmtree('data')
        make_npzs_signal()
        make_npzs_bkg()

    elif args.action == 'mergenpzs' or args.action == 'fromscratch':
        merge_npzs('data/train/raw')
        merge_npzs('data/test/raw')

    elif args.action == 'extrema':
        extrema('data/train/raw')



if __name__ == '__main__':
    main()