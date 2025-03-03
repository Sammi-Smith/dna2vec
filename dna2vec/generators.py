import logbook
import re
from Bio import SeqIO
from attic_util import util
from itertools import islice
import numpy as np

def remove_empty(str_list):
    return filter(bool, str_list)  # fastest way to remove empty string

class SeqCleaner:
    """
    Remove any ambiguous IUPAC nucleotide codes from a sequence, randomly replacing with an appropriate base
    """
    def __init__(self):
        pass

    #for a single sequence
    def get_acgt_seq(self, rng, seq):
        
        #convert string to a list, so we can make changes
        seq = list(seq.lower())

        for index, base in enumerate(seq):
            #for the standard 4 bases, don't change them
            if any(base == standard_base for standard_base in ['a', 'c', 'g', 't']):
                continue
            #for ambiguous IUPAC nucleotide codes, randomly replace with an appropriate base
            elif base == 'r':
                seq[index] = rng.choice(['a', 'g'])
            elif base == 'y':
                seq[index] = rng.choice(['c', 't'])
            elif base == 's':
                seq[index] = rng.choice(['g', 'c'])
            elif base == 'w':
                seq[index] = rng.choice(['a', 't'])
            elif base == 'k':
                seq[index] = rng.choice(['g', 't'])
            elif base == 'm':
                seq[index] = rng.choice(['a', 'c'])
            elif base == 'b':
                seq[index] = rng.choice(['c', 'g', 't'])
            elif base == 'd':
                seq[index] = rng.choice(['a', 'g', 't'])
            elif base == 'h':
                seq[index] = rng.choice(['a', 'c', 't'])
            elif base == 'v':
                seq[index] = rng.choice(['a', 'c', 'g'])
            elif base == 'n':
                seq[index] = rng.choice(['a', 'c', 'g', 't'])
            elif base == '.': #means gap
                seq[index] = rng.choice(['a', 'c', 'g', 't'])
            elif base == '-': #means gap
                seq[index] = rng.choice(['a', 'c', 'g', 't'])
            else:
                raise ValueError("Cannot convert mystery nucleotide '%s'" % base)


        #convert list back to string
        seq = "".join(seq).upper()

        return seq
    
    #for a list of sequences
    def get_acgt_seqs(self, rng, seqs):       
        return np.vectorize(self.get_acgt_seq)(rng, seqs)
    
class SeqFragmenter:
    """
    Split a sequence into small sequences based on some criteria, e.g. 'N' characters
    """
    def __init__(self):
        pass

    def get_acgt_seqs(self, seq):
        return remove_empty(re.split(r'[^ACGTacgt]+', str(seq)))

class SlidingKmerFragmenter:
    """
    Slide only a single nucleotide
    """
    def __init__(self, k_low, k_high):
        self.k_low = k_low
        self.k_high = k_high

    def apply(self, rng, seq):
        seq_len = len(seq)
        if seq_len < self.k_low: #this sequence is too short to be in the lookup table
            seq_to_add = ''.join(rng.choice(['A', 'C', 'G', 'T'], 
                                            size = self.k_low - seq_len))
            lengthened_seq = seq + seq_to_add
            return [lengthened_seq]
        elif seq_len < self.k_high: #this sequence is too short to be split into kmers
            return [seq]
        else:
            num_kmers = seq_len - self.k_high + 1
            rand_ints = rng.randint(self.k_low, self.k_high + 1, size = num_kmers)
            return [seq[i: i + rand_ints[i]] for i in range(num_kmers)]   
      
    def apply_to_list(self, rng, seqs):

        kmer_lists = []
        for seq in seqs:
            kmer_lists.append(self.apply(rng, seq))

        return kmer_lists

class DisjointKmerFragmenter:
    """
    Split a sequence into kmers
    """
    def __init__(self, k_low, k_high):
        self.k_low = k_low
        self.k_high = k_high

    @staticmethod
    def random_chunks(rng, li, min_chunk, max_chunk):
        """
        Both min_chunk and max_chunk are inclusive
        """
        it = iter(li)
        while True:
            head_it = islice(it, rng.randint(min_chunk, max_chunk + 1))
            nxt = '' . join(head_it)

            # throw out chunks that are not within the kmer range
            if len(nxt) >= min_chunk:
                yield nxt
            else:
                break

    def apply(self, rng, seq):
        seq = seq[rng.randint(self.k_low):]  # randomly offset the beginning to create more variations
        return list(DisjointKmerFragmenter.random_chunks(rng, seq, self.k_low, self.k_high))

class SeqMapper:
    def __init__(self, use_revcomp=True):
        self.use_revcomp = use_revcomp

    def apply(self, rng, seq):
        seq = seq.upper()
        if self.use_revcomp and rng.rand() < 0.5:
            return seq.reverse_complement()
        else:
            return seq

class SeqGenerator:
    def __init__(self, filenames, nb_epochs, seqlen_ulim=5000):
        self.filenames = filenames
        self.nb_epochs = nb_epochs
        self.seqlen_ulim = seqlen_ulim
        self.logger = logbook.Logger(self.__class__.__name__)
        self.logger.info('Number of epochs: {}'.format(nb_epochs))

    def filehandle_generator(self):
        for curr_epoch in range(self.nb_epochs):
            for filename in self.filenames:
                with open(filename) as file:
                    self.logger.info('Opened file: {}'.format(filename))
                    self.logger.info('Memory usage: {} MB'.format(util.memory_usage()))
                    self.logger.info('Current epoch: {} / {}'.format(curr_epoch + 1, self.nb_epochs))
                    yield file

    def generator(self, rng):
        for fh in self.filehandle_generator():
            # SeqIO takes twice as much memory than even simple fh.readlines()
            for seq_record in SeqIO.parse(fh, "fasta"):
                whole_seq = seq_record.seq
                self.logger.info('Whole fasta seqlen: {}'.format(len(whole_seq)))
                curr_left = 0
                while curr_left < len(whole_seq):
                    seqlen = rng.randint(self.seqlen_ulim // 2, self.seqlen_ulim)
                    segment = seq_record.seq[curr_left: seqlen + curr_left]
                    curr_left += seqlen
                    self.logger.debug('input seq len: {}'.format(len(segment)))
                    yield segment

class KmerSeqIterable:
    def __init__(self, rand_seed, seq_generator, mapper, seq_fragmenter, kmer_fragmenter, histogram):
        self.logger = logbook.Logger(self.__class__.__name__)
        self.seq_generator = seq_generator
        self.mapper = mapper
        self.kmer_fragmenter = kmer_fragmenter
        self.seq_fragmenter = seq_fragmenter
        self.histogram = histogram
        self.rand_seed = rand_seed
        self.iter_count = 0

    def __iter__(self):
        self.iter_count += 1
        rng = np.random.RandomState(self.rand_seed)
        for seq in self.seq_generator.generator(rng):
            seq = self.mapper.apply(rng, seq)
            acgt_seq_splits = list(self.seq_fragmenter.get_acgt_seqs(seq))
            self.logger.debug('Splits of len={} to: {}'.format(len(seq), [len(f) for f in acgt_seq_splits]))

            for acgt_seq in acgt_seq_splits:
                kmer_seq = self.kmer_fragmenter.apply(rng, acgt_seq)  # list of strings
                if len(kmer_seq) > 0:
                    if self.iter_count == 1:
                        # only collect stats on the first call
                        self.histogram.add(kmer_seq)
                    yield kmer_seq
