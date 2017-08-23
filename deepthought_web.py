import random
import string
import pickle
import cherrypy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
cherrypy.server.socket_host = '0.0.0.0'
cherrypy.config.update({'server.socket_port': 7071})
class DeepThought(object):

    def __init__(self):
        self.all_identifiers = pickle.load(open('all_identifiers.pkl'))
        self.X_tfidf = load_sparse_csr('x_tfidf.csr.npz')
        self.meta = pickle.load(open('meta.pkl'))
        self.tfidf_vect = pickle.load(open('tfidf_vect.pkl'))



    @cherrypy.expose
    def index(self):
        return """<html>
          <head></head>
          <body>
            <form method="get" action="arxiv_search">
              <input type="text" value="1207.4481" name="identifier" />
              <button type="submit">Similar Papers</button>
            </form>
            <form method="get" action="text_search">
              <input type="text" value="my astronomy paper" name="text" />
              <button type="submit">Similar Papers</button>
            </form>

          </body>
        </html>"""

    def _generate_table(self, ranked_similarity, ranked_identifiers):
        if np.sum(ranked_similarity) < 1e-10: return "No matches found"
        print ranked_similarity, ranked_identifiers
        j = 0

        table_similarity = []
        table_identifier = []
        table_title = []
        table_link = []
        for simil, identifier in zip(ranked_similarity, ranked_identifiers):
            table_similarity.append(simil)
            table_identifier.append(identifier)
            if identifier in self.meta:
                table_title.append(self.meta[identifier]['title'])
            else:
                table_title.append('Title N/A')
            if '.' in identifier:
                table_link.append('https://arxiv.org/abs/{0}'.format(identifier))
            else:
                table_link.append('https://arxiv.org/abs/astro-ph/{0}'.format(identifier[8:]))
            j+=1
            print 'at', j
            if j > 50:
                break
        data_table = pd.DataFrame(zip(table_identifier, table_title, table_link, table_similarity),
                                    columns = ['identifier', 'title', 'link', 'similarity'])
        return data_table.to_html()

    def _get_similar_documents(self, test_document):
        similarity = np.squeeze((self.X_tfidf * test_document.T).A)
        similarity_argsort = np.argsort(similarity)[::-1]
        ranked_similarity = similarity[similarity_argsort]
        ranked_identifiers = np.array(self.all_identifiers)[similarity_argsort]
        return ranked_similarity, ranked_identifiers
    @cherrypy.expose
    def arxiv_search(self, identifier='1207.4481'):
        if identifier not in self.all_identifiers:
            return "unknown identifier {0}".format(identifier)
        else:
            test_document_id = self.all_identifiers.index(identifier)
            test_document = self.X_tfidf[test_document_id]
            ranked_similarity, ranked_identifiers = self._get_similar_documents(test_document)

            return self._generate_table(ranked_similarity, ranked_identifiers)
        #return ''.join(random.sample(string.hexdigits, int(length)))
    @cherrypy.expose
    def text_search(self, text='astronomy galaxy star'):
        test_document = self.tfidf_vect.transform([text])
        ranked_similarity, ranked_identifiers = self._get_similar_documents(test_document)

        return self._generate_table(ranked_similarity, ranked_identifiers)



def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

if __name__ == '__main__':
    print 'loading...'
    dt = DeepThought()
    print "loading done"
    cherrypy.quickstart(dt)
