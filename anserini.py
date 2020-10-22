from pyserini.setup import configure_classpath
configure_classpath('/home/azahrayi/memory-augmentation/anserini/target')
from jnius import autoclass
JString = autoclass('java.lang.String')
JList = autoclass('java/util/ArrayList')
JSimpleSearcher = autoclass('io.anserini.search.SimpleSearcher')

analyzerUtils = autoclass('io.anserini.analysis.AnalyzerUtils')
englishAnalyzer = autoclass('io.anserini.analysis.EnglishStemmingAnalyzer')

chararrayset = autoclass('org.apache.lucene.analysis.CharArraySet')
stopwords = chararrayset.EMPTY_SET
engAnalyzer = englishAnalyzer(stopwords)  # no stop words are passed


JIndexReaderUtils = autoclass('io.anserini.index.IndexReaderUtils')
JTerm = autoclass("org.apache.lucene.index.Term")
stopwords_temp = ('kiki_snow_puppy123', 'robashow', 'esementera', 'hotel.locked', 'this', 'keepingsecrets99', 'spaghetti_monster123', 'to', 'e462a86ef33a1a57', 'scfsf_zps9d5a5806', 'womensrightsi', 'mackerrrrrral', 'falutinest', 'spaceship.that', 'and', 'of', 'dancable.would', 'or', 'intriqte', 'whipped.ive', 'relmfu', 'sphotos.ak.fbcdn.net', 'fluids.png', 'bokuriku', 'mcfista', 'they', 'lerman_logan_bill09', 'are', 'an', 'id_aks9obaw', 'files.broadsheet.ie', 'sawpenguins', 'meaganzhott', '____things.com', 'demonters', 'no', '9dhgzo38', 'sc003193fd.jpg', 'l_65e797152557b6e27bcc5da24b14b490', 'anemone_jhade13', 'arealityshowthatturnsouttobeahostagetrap', 'feoncai', 'amytivelle', 'is', '9db3noaurcy', 'clap8x', 'their', "headq'ters", 'isk8board327', 'iburieded', 'was', '37565_1231894736671_1808244714_457529_2225326_n.jpg', 'blzzfigg', '1290573949583', "s'acapella", 'line:its', 'by5gyfixc', 'in', 'it', 'then', 'n5gyxenbiho', 'forgeeeeettt', 'knowsooo', 'for', 'by', 'that', 'everhthings', 'will', 'hardjackets', 'into', 'with', '1humanastronaut', 'these', '9gainvqclma', 'tell_w0qqfnuz1qqfsooz1qqfsopz3qqsatitlezshowq20q27nq20tellqqxpufuzx', 'on', 'contamered', 'karkaroff_i', 'kzqyijtwrgq', 'but', 'ngenradio.com', 'not', 'perspectives:one', '2blerman', '650735171_e6db3d4c28_o.jpg', 'maroco3uo.jpg', 'ss.tf', 'chlorhes', 'mingjosly', 'bahtovin', 'if', 'raper.the', 'tiggercline', 'sry4bad', 'as', 'recolletcs', 'poet.would', 'the', 'there', 'ponpocky', 'be', 'such', 'rose_duhh', 'endijiyfjnm', 'a', 'at')

import re

def tokenizeString(string, analyzer=None):
    if analyzer == 'lucene':
        return analyzerUtils.tokenize(engAnalyzer, string).toArray()
    return re.split('\W+', string.lower())

searcher = JSimpleSearcher(JString('/GW/D5data-10/Clueweb/anserini0.9-index.clueweb09.englishonly.nostem.stopwording'))
reader = JIndexReaderUtils.getReader(JString(
    '/home/azahrayi/memory-augmentation/query-simulation/ForgetCorpus/webis-docs/index.pos+docvectors+raw'))

minisearcher = JSimpleSearcher(JString(
    '/home/azahrayi/memory-augmentation/query-simulation/ForgetCorpus/webis-docs/index.pos+docvectors+raw+porter'))
minireader = JIndexReaderUtils.getReader(JString(
    '/home/azahrayi/memory-augmentation/query-simulation/ForgetCorpus/webis-docs/index.pos+docvectors+raw'))


def get_term_coll_freq(term):
    jterm = JTerm("contents", term.lower())
    cf = reader.totalTermFreq(jterm)
    return cf


def get_term_doc_freq(term):
    jterm = JTerm("contents", term)
    df = reader.docFreq(jterm)
    return df