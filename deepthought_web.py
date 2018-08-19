import random
import string
import pickle
import cherrypy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import re
import os
from jinja2 import Environment, FileSystemLoader

path = os.path.abspath(os.path.dirname(__file__))
config = {
  'global' : {
    'tools.proxy.on':True,
    'server.socket_host' : '0.0.0.0',
    'server.socket_port' : 7071,
    'server.thread_pool' : 8
  },
  '/' : {'tools.staticdir.root':path},
  '/css' : {
    'tools.staticdir.on'  : True,
    'tools.staticdir.dir' : os.path.join(path, 'css')
  },
  '/fonts' : {
    'tools.staticdir.on'  : True,
    'tools.staticdir.dir' : os.path.join(path, 'fonts')
  }
}
env = Environment(loader=FileSystemLoader(os.path.join(path, 'templates')))


class SKTFIDFCompare(object):
    
    similarity = None
    
    @classmethod
    def from_hdf(cls, fname):
        meta = pd.read_hdf(fname, 'meta')
        df_matrix = pd.read_hdf(fname, 'tfidf_matrix')
        matrix = sparse.coo_matrix((df_matrix.tfidf.values, 
                                    (df_matrix.row, df_matrix.col.values))).tocsr()
        vocabulary = pd.read_hdf(fname, 'vocabulary')
        return cls(matrix, vocabulary, meta)
    
    def __init__(self, matrix, vocabulary, meta):
        self.matrix = matrix
        self.vocabulary = vocabulary
        self.meta = meta
    
    def get_doc_vector(self, article_id):
        self.cur_article_id = article_id
        paper_id = self.meta.index.get_loc(article_id)
        doc_vector = self.matrix[paper_id]
        self.vocabulary['cur_word_weight'] = np.squeeze(doc_vector.A)
        return doc_vector

    def compare_paper(self, article_id):
        self.cur_article_id = article_id
        doc_vector = self.get_doc_vector(article_id)
        self.similarity = np.squeeze((self.matrix * doc_vector.T).A)
        self.ranked_similarity = np.argsort(self.similarity)[::-1]
        return self.similarity
        #


class DeepThought(object):

    DATASET_FNAME = 'dt_201806_tfidf_skl.h5'
    def __init__(self):
        print('Loading Dataset')
        self.dt_tfidf = SKTFIDFCompare.from_hdf(self.DATASET_FNAME)
        print('Loaded Dataset')

    @cherrypy.expose
    def index(self):
        template = env.get_template('index.html')
        return template.render()


    @cherrypy.expose
    def arxiv_search(self, identifier='1207.4481'):
        identifier = identifier.strip()
        template = env.get_template('arxiv_search')
        if identifier not in self.dt_tfidf.meta.index:
            return template.render(identifier=identifier, unknown_id=True)
        else:
            similarity = self.dt_tfidf.compare_paper(identifier)
            #test_document_id = self.dt_tfidf.meta.index.getloc(identifier)
            #test_document = self.X_tfidf[test_document_id]
            #ranked_similarity, ranked_identifiers = self._get_similar_documents(test_document)
            
            data_table = self.dt_tfidf.meta.copy().iloc[self.dt_tfidf.ranked_similarity]
            data_table['similarity'] = similarity[self.dt_tfidf.ranked_similarity]
            data_table['identifier'] = data_table.index
            data_table['link'] = ['https://arxiv.org/abs/{0}'.format(identifier) for identifier in data_table['identifier']]

            return template.render(identifier=identifier, data_table=data_table.iloc[:50].to_dict('records'))
        #return ''.join(random.sample(string.hexdigits, int(length)))
    

    #@cherrypy.expose
    #def text_search(self, text='astronomy galaxy star'):
    #    test_document = self.tfidf_vect.transform([text])
    #    ranked_similarity, ranked_identifiers = self._get_similar_documents(test_document)
    #    data_table = self._generate_table(ranked_similarity, ranked_identifiers)

        #template = env.get_template('arxiv_search')
        #return template.render(search_str=text, data_table=data_table)


    def _generate_table(self, ranked_similarity, ranked_identifiers):
        if np.sum(ranked_similarity) < 1e-10: return "No matches found"
        print (ranked_similarity, ranked_identifiers)
        j = 0

        table_similarity = []
        table_identifier = []
        table_title = []
        table_link = []
        for simil, identifier in zip(ranked_similarity, ranked_identifiers):
            table_similarity.append(simil)
            # older ID's look like "astro-ph0410673"; they are more useful with the slash
            identifier = identifier if '.' in identifier else '{}/{}'.format(identifier[:8], identifier[8:])
            table_identifier.append(identifier)
            if identifier in self.meta:
                title = str(self.meta[identifier]['title'])
                # strip brackets etc.:
                title = re.sub(r'(\[\'|\[\"|\'\]|\"\]|\\n|\n)', "", title)
                table_title.append(title)
            else:
                table_title.append('Title N/A')
            table_link.append('https://arxiv.org/abs/{0}'.format(identifier))
            j+=1
            print('at', j)
            if j > 50:
                break
        data_table = pd.DataFrame(zip(table_identifier, table_title, table_link, table_similarity),
                                    columns = ['identifier', 'title', 'link', 'similarity'])
        return data_table.to_dict('records')


    def _get_similar_documents(self, test_document):
        similarity = np.squeeze((self.X_tfidf * test_document.T).A)
        similarity_argsort = np.argsort(similarity)[::-1]
        ranked_similarity = similarity[similarity_argsort]
        ranked_identifiers = np.array(self.all_identifiers)[similarity_argsort]
        return ranked_similarity, ranked_identifiers

dt_app = DeepThought()
cherrypy.quickstart(dt_app, '/deepthought', config=config)

