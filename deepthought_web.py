import random
import string
import pickle
import cherrypy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import re
import os
from jinja2 import Environment, FileSystemLoader

path = os.path.abspath(os.path.dirname(__file__))
config = {
  'global' : {
    'server.socket_host' : '0.0.0.0',
    'server.socket_port' : 7071,
    'server.thread_pool' : 8
  },
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

class DeepThought(object):

    def __init__(self):
        return
        self.all_identifiers = pickle.load(open('all_identifiers.pkl'))
        self.X_tfidf = load_sparse_csr('x_tfidf.csr.npz')
        self.meta = pickle.load(open('meta.pkl'))
        self.tfidf_vect = pickle.load(open('tfidf_vect.pkl'))


    @cherrypy.expose
    def index(self):
        template = env.get_template('index.html')
        return template.render()


    @cherrypy.expose
    def arxiv_search(self, identifier='1207.4481'):
        identifier = identifier.strip()
        template = env.get_template('arxiv_search')
        if identifier not in self.all_identifiers:
            return template.render(identifier=identifier, unknown_id=True)
        else:
            test_document_id = self.all_identifiers.index(identifier)
            test_document = self.X_tfidf[test_document_id]
            ranked_similarity, ranked_identifiers = self._get_similar_documents(test_document)
            data_table = self._generate_table(ranked_similarity, ranked_identifiers)

            return template.render(identifier=identifier, data_table=data_table)
        #return ''.join(random.sample(string.hexdigits, int(length)))
    

    @cherrypy.expose
    def text_search(self, text='astronomy galaxy star'):
        test_document = self.tfidf_vect.transform([text])
        ranked_similarity, ranked_identifiers = self._get_similar_documents(test_document)
        data_table = self._generate_table(ranked_similarity, ranked_identifiers)

        template = env.get_template('arxiv_search')
        return template.render(search_str=text, data_table=data_table)


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


def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

dt = DeepThought()
cherrypy.quickstart(dt, '/', config)

#cherrypy.tree.mount(my_controller.Root(), script_name=´´, config=app_conf)
#if __name__ == '__main__':
#    print('loading...')

#    print("loading done")
    # cherrypy.quickstart(dt)
