import logging
import pickle
from pathlib import Path

import numpy as np
import utils.sourmash as sourmash
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import utils.metagenome_sim as metasim
from utils.cache import KeyedCache
from utils.genome_utils import GenomeTools
from utils.taxonomy_utils import TaxonomyTools
from utils.transforms import ToClassNumbers, SummarizeTargets
from utils.util import (check_file_integrity, download_file, ensure_dir,
                        fetch_url, strip_header)


class RefSeqProkaryotaBags(Dataset):
    def __init__(self,
                 bag_size,
                 total_bags,
                 target_format,
                 genome_dir,
                 taxonomy_dir,
                 accessions_file=None,
                 taxids_list=None,
                 download=False,
                 ncbi_email='your-email@domain.com',
                 ncbi_api=None,
                 transform_x=None,
                 transform_y=None,
                 filter_by_level=None,
                 num_to_keep=1,
                 dataset_distribution='uniform',
                 reseed_every_n_bags=1,
                 genome_cache_size=1000,
                 num_workers=0,
                 single_read_target_vectors=False):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.bag_size = bag_size
        self.total_bags = total_bags
        self._num_workers = num_workers
        self.reseed_every_n_bags = reseed_every_n_bags
        self.single_read_target_vectors = single_read_target_vectors

        # Initialize instance level dataset
        self.instance_dataset = RefSeqProkaryota(
            genome_dir=genome_dir,
            taxonomy_dir=taxonomy_dir,
            total_samples=self.bag_size,
            accessions_file=accessions_file,
            taxids_list=taxids_list,
            download=download,
            ncbi_email=ncbi_email,
            ncbi_api=ncbi_api,
            transform_x=transform_x,
            transform_y=transform_y,
            filter_by_level=filter_by_level,
            num_to_keep=num_to_keep,
            dataset_distribution=dataset_distribution,
            genome_cache_size=genome_cache_size)

        # Initialize bag loader
        self.bag_loader = DataLoader(self.instance_dataset,
                                     batch_size=self.bag_size,
                                     shuffle=False,
                                     sampler=None,
                                     batch_sampler=None,
                                     num_workers=0,
                                     collate_fn=None,
                                     pin_memory=True)

        self.set_to_lognorm()
        self.set_to_uniform()
        self.set_to_default_sim()
        self.idx_offset = 0

        self.trsfm_y = SummarizeTargets(self.instance_dataset.rank_sizes,
                                        target_format=target_format,
                                        eps=0.001)

    def _get_bag(self, idx):
        self.enable_multithreading_if_possible()
        if idx % self.reseed_every_n_bags == 0:
            self.instance_dataset.simulator.set_seed(idx //
                                                     self.reseed_every_n_bags)
        self.instance_dataset.idx_offset = idx * self.bag_size
        return next(iter(self.bag_loader))

    def __len__(self):
        return self.total_bags

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError('{} is >= {}'.format(idx, len(self)))

        idx += self.idx_offset

        x, y = self._get_bag(idx)

        if not self.single_read_target_vectors:
            y = self.trsfm_y(y)

        return x, y

    def enable_multithreading_if_possible(self):
        if self.instance_dataset.genome_cache_is_full():
            if self.bag_loader.num_workers != self._num_workers:
                self.bag_loader.num_workers = self._num_workers
                self.logger.info(f'Enabling {self.bag_loader.num_workers} '
                                 'workers for data loading...')

    def set_to_lognorm(self):
        self.instance_dataset.set_to_lognorm()

    def set_to_uniform(self):
        self.instance_dataset.set_to_uniform()

    def set_to_default_sim(self):
        self.instance_dataset.set_to_default_sim()


class RefSeqProkaryota(Dataset):
    """Prokaryota from RefSeq database."""

    # REFSEQ_URL = ('ftp://ftp.ncbi.nlm.nih.gov'
    #               '/genomes/refseq/assembly_summary_refseq.txt')
    PROK_URL = ('ftp://ftp.ncbi.nlm.nih.gov'
                '/genomes/GENOME_REPORTS/prokaryotes.txt')
    TAX_URL = ('ftp://ftp.ncbi.nlm.nih.gov' '/pub/taxonomy/taxdump.tar.gz')

    def __init__(self,
                 genome_dir,
                 taxonomy_dir,
                 total_samples,
                 accessions_file=None,
                 taxids_list=None,
                 download=True,
                 ncbi_email='your-email@domain.com',
                 ncbi_api=None,
                 transform_x=None,
                 transform_y=None,
                 filter_by_level=None,
                 num_to_keep=1,
                 dataset_distribution='uniform',
                 genome_cache_size=1000):
        """
        Args:
            genome_dir (string): Path to the folder containing
                all genome files(or where to store it).
            taxonomy_dir (string): Path to the folder that will store
                taxonomy data.
            total_samples (integer): Number of samples to simulate a fixed size
                dataset.
            rmin (integer, optional): Minimum read length
            rmax (integer, optional): Maximum read length
            accessions_file (string, optional): Path to file with accessions
                IDs. If not provided, all genomes in genome_dir will be used.
            download (bool, optional): Whether to download the dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
            genome_cache_size (integer, optional): Number of genomes to store
                in memory. -1 for unlimited size
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.genome_dir = Path(genome_dir)
        self.taxonomy_dir = Path(taxonomy_dir)
        self.total_samples = total_samples
        if accessions_file:
            self.accessions_file = Path(accessions_file)
        else:
            self.accessions_file = None
        self.taxids_list = taxids_list
        self.transform_x = transform_x
        self.transform_y = transform_y

        self.filter_by_level = filter_by_level
        self.num_to_keep = num_to_keep

        self.genome_cache = KeyedCache(maxsize=genome_cache_size)
        self.taxonomy = None
        self.gt = GenomeTools(ncbi_email, ncbi_api)
        self.tt = TaxonomyTools(ncbi_email, ncbi_api)
        self.random = np.random.RandomState(seed=0)
        self.tcn = ToClassNumbers()
        self.final_entries = []
        self.class_percentages = []
        self.rank_sizes = []
        self.categories = [
            'phylum', 'class', 'order', 'family', 'genus', 'species'
        ]

        self.idx_offset = 0

        metasim_options = ['uniform', 'lognormal']
        dist, level = self._check_metasim_options(dataset_distribution)
        if dist not in metasim_options:
            raise AssertionError(
                'dataset_distribution can only be uniform or lognormal')
        self.dataset_distribution = dist
        self.distribution_level = level
        self.uniform_sim = None
        self.lognormal_sim = None
        self.simulator = None

        if download is True:
            self._download()
        self._check()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError('{} is >= {}'.format(idx, len(self)))

        idx += self.idx_offset

        # Randomly choose genome
        id_idx = self.simulator.get_accession_index(idx)
        id_path = self.final_entries[id_idx]
        id = id_path.stem

        # Use idx to deterministically choose genome
        # id_idx = idx % len(self.final_entries)
        # id_path = self.final_entries[id_idx]
        # id = id_path.stem

        # Load
        try:
            genome = self.genome_cache[id]
        except KeyError:
            genome = self.gt.read_fasta(id_path)
            self.genome_cache[id] = genome

        x = genome
        y = self.taxonomy[id_idx]

        if self.transform_x:
            x = self.transform_x(x, idx=idx)
        if self.transform_y:
            y = self.transform_y(y, idx=idx)

        return (x, y)

    def _load_tax_tree(self):
        try:
            self.logger.info("Loading taxonomic tree...")
            tree = self.tt.load_taxfile(
                str(self.taxonomy_dir / 'taxdump.tar.gz'))
        except FileNotFoundError:
            self.logger.error(
                'Taxonomic data missing..please run download first...')
            exit(1)
        return tree

    def _download_all_prok_accessions(self):
        self.logger.info('Downloading data from RefSeq database...')
        prokaryotes = fetch_url(self.PROK_URL)
        if prokaryotes is None:
            self.logger.error(
                "Could not download list of prokaryota..Please try again")
            exit(1)
        prokaryotes = strip_header(prokaryotes.split('\n'), '#')

        all_accessions = []
        for line in prokaryotes:
            line = line.split('\t')
            genomes = line[8]
            # reftype = line[19]
            if genomes != '-':
                all_accessions.extend([
                    entry.split(':')[1].split('/')[0]
                    for entry in genomes.split(';')
                    if len(entry.split(':')) > 1
                ])
        return all_accessions

    def _download_accessions_by_taxid(self, tax_ids, tax_tree):
        self.logger.info(f"Collecting leaves from taxonomy ids: {tax_ids}...")
        leaves = []
        for t in tax_ids:
            leaves.extend(tax_tree.get_all_leaves(t))
        self.logger.info(f"{len(leaves)} tax ids found..")

        all_accessions = []
        self.logger.info("Collecting accession numbers...")
        for taxid in tqdm(leaves, total=len(leaves)):
            assembly_id = self.tt.get_assembly_id_from_tax_id(taxid)
            if assembly_id:
                if self.tt.assembly_has_complete_genome(assembly_id):
                    accessions = self.tt.get_accessions_from_assembly_id(
                        assembly_id)
                    all_accessions.extend([
                        acc for acc in accessions
                        if self.tt.is_genome_from_chromosome(acc)
                    ])
        return all_accessions

    def _load_accessions_from_file(self, file):
        file = Path(file)
        self.logger.info("Loading {}...".format(file))
        with file.open('r') as f:
            all_accessions = list(set([line.strip() for line in f]))
        return all_accessions

    def _download_taxonomy(self):
        ensure_dir(self.taxonomy_dir)
        self.logger.info("Downloading taxonomic tree...")
        res = download_file(self.TAX_URL, self.taxonomy_dir / 'taxdump.tar.gz')
        if res is None:
            self.logger.error(
                "Could not download taxdump information...Please try again")
            exit(1)

    def _load_acc2taxid(self):
        try:
            with (self.taxonomy_dir / 'acc2taxid.pkl').open('rb') as f:
                acc2taxid = pickle.load(f)
        except Exception:
            acc2taxid = {}
        return acc2taxid

    def _save_acc2taxid(self, acc2taxid):
        with (self.taxonomy_dir / 'acc2taxid.pkl').open('wb') as f:
            pickle.dump(acc2taxid, f)

    def _filter_accessions_with_taxids(self, accessions, taxids, tax_tree,
                                       acc2taxid):
        leaves = []
        for tid in taxids:
            leaves.extend(tax_tree.get_all_leaves(tid))

        new_accessions = []
        for acc in accessions:
            try:
                if acc2taxid[acc] in leaves:
                    new_accessions.append(acc)
            except KeyError:
                self.logger.warning(("Could not find TaxId for"
                                     " entry {}..skipping...").format(acc))

        return new_accessions

    def _download(self):

        self._download_taxonomy()

        # Downloading IDs from RefSeq/load from file/download family
        if self.taxids_list:
            tree = self._load_tax_tree()
            accessions = self._download_accessions_by_taxid(
                self.taxids_list, tree)
        elif self.accessions_file:
            accessions = self._load_accessions_from_file(self.accessions_file)
        else:
            accessions = self._download_all_prok_accessions()

        accessions = [id.split('.')[0] for id in accessions]
        accessions.sort()

        self.logger.info("Downloading TaxIds...")
        acc2taxid = self._load_acc2taxid()
        for i, id in tqdm(enumerate(accessions), total=len(accessions)):
            try:
                taxid = acc2taxid[id]
            except KeyError:
                taxid = self.tt.get_tax_id(id)
            if not taxid:
                self.logger.warning(("Could not find TaxId for"
                                     " entry {}..skipping...").format(id))
                accessions.pop(i)
            else:
                acc2taxid[id] = taxid
            if i % 100 == 0:  # Save every 100 iterations
                self._save_acc2taxid(acc2taxid)

        self._save_acc2taxid(acc2taxid)

        self.logger.info("Downloading genomes...")
        ensure_dir(self.genome_dir)
        for id in tqdm(accessions, total=len(accessions)):
            self.gt.download_genome(id, genome_dir=self.genome_dir)

    def _check(self):
        """
        Check if the genomes have been processed and an index has been created
        Create index if index is not detected
        """
        acc2taxid = self._load_acc2taxid()
        tree = self._load_tax_tree()
        self.acc2taxid = acc2taxid

        self.logger.info("Checking and filtering downloaded files...")

        if self.accessions_file:
            accessions = self._load_accessions_from_file(self.accessions_file)
        else:
            accessions = set([f.stem for f in self.genome_dir.glob('*.fasta')])

        accessions = [id.split('.')[0] for id in accessions]
        accessions.sort()

        if self.taxids_list:
            accessions = self._filter_accessions_with_taxids(
                accessions, self.taxids_list, tree, acc2taxid)

        tree.trim_tree([acc2taxid[acc] for acc in accessions],
                       levels_to_keep=self.categories)
        tree.ensure_levels_exist(ordered_level_names=self.categories)

        if self.filter_by_level:

            # Choose num_to_keep nodes from level filter_by_level
            # such that leaves are maximized
            ids_at_level = np.array(tree.get_ids_at_level(
                self.filter_by_level))
            num_of_leaves_per_id = [
                len(tree.get_all_leaves(id)) for id in ids_at_level
            ]
            tax_ids_chosen = ids_at_level[np.argsort(num_of_leaves_per_id)
                                          [::-1][:self.num_to_keep]]

            self.logger.info(
                f"Filtering accessions using TaxIDs: {tax_ids_chosen}")
            accessions = self._filter_accessions_with_taxids(
                accessions, tax_ids_chosen, tree, acc2taxid)
            accessions.sort()

        self.final_entries = []
        tax_ids_used = set()
        for id in tqdm(accessions, total=len(accessions)):
            fasta_file = self.genome_dir / (id + '.fasta')
            md5file = self.genome_dir / (id + '.fasta.md5')
            if not check_file_integrity(fasta_file, md5file):
                self.logger.warning(
                    "Cannot verify integrity of file {}..skipping..".format(
                        fasta_file.name))
            elif fasta_file.stem not in acc2taxid:
                self.logger.warning(
                    "Cannot find tax id of file {}..skipping..".format(
                        fasta_file.name))
            elif acc2taxid[fasta_file.stem] in tax_ids_used:
                self.logger.warning(
                    "Multiple genomes with same tax id.. skipping {}..".format(
                        fasta_file.name))
            else:
                self.final_entries.append(fasta_file)
                tax_ids_used.add(acc2taxid[fasta_file.stem])

        tree.trim_tree([acc2taxid[acc.stem] for acc in self.final_entries],
                       levels_to_keep=self.categories)
        tree.ensure_levels_exist(ordered_level_names=self.categories)

        self.logger.info("{} genome entries found".format(
            len(self.final_entries)))

        # self.logger.info('Calculating minhashes of dataset...')
        # sourmash.create_signatures(self.final_entries, ksize=17, verbose=True)

        # self.logger.info('Calculating average jaccard similarity..')
        # dist = sourmash.get_average_dist(self.final_entries)
        # self.logger.info(f'Average jaccard similarity index: {dist}')

        taxonomy = np.zeros(
            (len(self.final_entries), len(self.categories) + 1), dtype=np.int)
        for i, id in enumerate(self.final_entries):
            id = id.stem
            taxonomy[i] = tree.collect_path(acc2taxid[id], self.categories)
        self.tree = tree
        taxonomy = self.tcn(taxonomy)

        temp_cats = [*self.categories, 'leaf']
        class_percentages = [None] * taxonomy.shape[1]
        rank_sizes = [None] * taxonomy.shape[1]
        last_none = -1
        for i in range(0, taxonomy.shape[1]):
            if len(self.tcn.counts[i]) == 1:
                last_none = i
                continue
            class_percentages[i] = np.true_divide(self.tcn.counts[i],
                                                  taxonomy.shape[0])
            rank_sizes[i] = self.tcn.counts[i].shape[0]
            self.logger.info("{}: {}".format(temp_cats[i], rank_sizes[i]))

        self.class_percentages = class_percentages[last_none + 1:]
        self.rank_sizes = rank_sizes[last_none + 1:]
        self.categories = self.categories[last_none + 1:]
        self.categories_with_leaf = [*self.categories, 'leaf']
        self.taxonomy = taxonomy[:, last_none + 1:]
        self.tcn.set_offset(last_none + 1)

        self.set_to_default_sim()

    def _check_metasim_options(self, option):
        option = option.split('_')
        if len(option) == 1:
            return option[0], None
        return option[0], option[1]

    def set_to_uniform(self):
        if not self.uniform_sim:
            distribution_level = None
            if self.dataset_distribution == 'uniform':
                distribution_level = self.distribution_level
            if distribution_level:
                self.uniform_sim = metasim.UniformAtLevelSimulator(
                    self.final_entries,
                    self.acc2taxid,
                    self.tree,
                    self.categories,
                    start_level=distribution_level)
            else:
                self.uniform_sim = metasim.UniformSimulator(self.final_entries)
        self.simulator = self.uniform_sim

    def set_to_lognorm(self):
        if not self.lognormal_sim:
            distribution_level = None
            if self.dataset_distribution == 'lognormal':
                distribution_level = self.distribution_level
            if not distribution_level:
                distribution_level = 'genus'
            self.lognormal_sim = metasim.LogNormalSimulator(
                self.final_entries,
                self.acc2taxid,
                self.tree,
                self.categories,
                start_level=distribution_level,
                max_strains_per_id=10)
        self.simulator = self.lognormal_sim

    def set_to_default_sim(self):
        if self.dataset_distribution == 'uniform':
            self.set_to_uniform()
        elif self.dataset_distribution == 'lognormal':
            self.set_to_lognorm()

    def genome_cache_is_full(self):
        if len(self.genome_cache.data) == self.genome_cache.maxsize:
            return True
        if len(self.genome_cache.data) == len(self.final_entries):
            return True
        return False


if __name__ == '__main__':
    # rsp = RefSeqProkaryota(
    #     ("/home/ageorgiou/eth/spring2019/thesis"
    #      "/data/refseq_prokaryota/genomes"),
    #     ("/home/ageorgiou/eth/spring2019/thesis/"
    #      "data/refseq_prokaryota/taxonomy"),
    #     total_samples=10000,
    #     accessions_file=('/home/ageorgiou/eth/spring2019/thesis/data/'
    #                      'refseq_prokaryota/ncbi_id_training.txt'),
    #     download=False)

    import utils.transforms as mytransforms
    import torch
    g2read = mytransforms.GenomeToNoisyKmerRead(
        '/home/ageorgiou/eth/spring2019/thesis/data/refseq_prokaryota/token_8mer',
        "perfect_16",
        0,
        0,
        p=0,
        forward_reads_only=False)
    trsfm_x = mytransforms.Compose(
        [g2read,
         mytransforms.ToTensorWithView(dtype=torch.long, view=[-1])])
    trsfm_y = mytransforms.Compose([mytransforms.ToTensorWithView()])
    rsp = RefSeqProkaryotaBags(
        bag_size=4,
        total_bags=2,
        target_format='probs',
        genome_dir=("/home/ageorgiou/eth/spring2019/thesis"
                    "/data/refseq_prokaryota/genomes"),
        taxonomy_dir=("/home/ageorgiou/eth/spring2019/thesis/"
                      "data/refseq_prokaryota/taxonomy"),
        accessions_file=('/home/ageorgiou/eth/spring2019/thesis/data/'
                         'refseq_prokaryota/ncbi_id_training_filtered.txt'),
        taxids_list=None,
        download=False,
        transform_x=trsfm_x,
        transform_y=trsfm_y,
        dataset_distribution='lognormal',
        genome_cache_size=10,
        num_workers=2)

    import time

    start = time.time()
    for i in range(2):
        temp = rsp[i]
        print(temp)
        print()
    print(time.time() - start)

    # rsp.

    # Check if cache is making a difference
    # start = time.time()
    # for i in range(1000):
    #     temp = rsp[i]
    # print(time.time() - start)
