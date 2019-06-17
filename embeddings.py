from tqdm import tqdm
from annoy import AnnoyIndex
import numpy as np

class PreTrainedEmbeddings(object):
    """ A wrapper around pre-trained word vectors and their use """
    def __init__(self, word_to_index, word_vectors):
        """
        Args:
            word_to_index (dict): mapping from word to integers
            word_vectors (list of numpy arrays)
        """
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}

        self.index = AnnoyIndex(len(word_vectors[0]), metric='euclidean')
        print("Building Index!")
        for _, i in self.word_to_index.items():
            self.index.add_item(i, self.word_vectors[i])
        self.index.build(50)
        print("Finished!")
        
    @classmethod
    def from_embeddings_file(cls, embedding_file):
        """Instantiate from pre-trained vector file.
        
        Vector file should be of the format:
            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N
        
        Args:
            embedding_file (str): location of the file
        Returns: 
            instance of PretrainedEmbeddigns
        """
        word_to_index = {}
        word_vectors = []

        with open(embedding_file) as fp:
            for line in fp.readlines():
                line = line.split(" ")
                word = line[0]
                vec = np.array([float(x) for x in line[1:]])
                
                word_to_index[word] = len(word_to_index)
                word_vectors.append(vec)
                
        return cls(word_to_index, word_vectors)
    
    def get_embedding(self, word):
        """
        Args:
            word (str)
        Returns
            an embedding (numpy.ndarray)
        """
        return self.word_vectors[self.word_to_index[word]]

    def get_closest_to_vector(self, vector, n=1):
        """Given a vector, return its n nearest neighbors
        
        Args:
            vector (np.ndarray): should match the size of the vectors 
                in the Annoy index
            n (int): the number of neighbors to return
        Returns:
            [str, str, ...]: words that are nearest to the given vector. 
                The words are not ordered by distance 
        """
        nn_indices = self.index.get_nns_by_vector(vector, n)
        return [self.index_to_word[neighbor] for neighbor in nn_indices]
    
    def compute_and_print_analogy(self, word1, word2, word3):
        """Prints the solutions to analogies using word embeddings

        Analogies are word1 is to word2 as word3 is to __
        This method will print: word1 : word2 :: word3 : word4
        
        Args:
            word1 (str)
            word2 (str)
            word3 (str)
        """
        vec1 = self.get_embedding(word1)
        vec2 = self.get_embedding(word2)
        vec3 = self.get_embedding(word3)

        # now compute the fourth word's embedding!
        spatial_relationship = vec2 - vec1
        vec4 = vec3 + spatial_relationship

        closest_words = self.get_closest_to_vector(vec4, n=4)
        existing_words = set([word1, word2, word3])
        closest_words = [word for word in closest_words 
                             if word not in existing_words] 

        if len(closest_words) == 0:
            print("Could not find nearest neighbors for the computed vector!")
            return
        
        for word4 in closest_words:
            print("{} : {} :: {} : {}".format(word1, word2, word3, word4))
    
    def get_batch(self, batch_size, qmax, amax, df):
        q_input = np.ones((batch_size, qmax), dtype=int)
        a1_input = np.ones((batch_size, amax), dtype=int)
        wa1_input = np.ones((batch_size, amax), dtype=int)
        rand_ind = np.random.randint(len(df),size=batch_size)
        df_batch = df.loc[rand_ind]
        q_aux=df_batch["q1"].values
        len_q = df_batch["len_q1"].values
        
        a1_aux=df_batch["a1"].values
        len_a1 = df_batch["len_a1"].values
        
        aux1=np.random.randint(0,3)
        if (aux1==0):
            wa1_aux=df_batch["wa1"].values
            len_wa1=df_batch["len_wa1"].values
        elif (aux1==1):
            wa1_aux=df_batch["wa2"].values
            len_wa1=df_batch["len_wa2"].values
        elif (aux1==2):
            wa1_aux=df_batch["wa3"].values
            len_wa1=df_batch["len_wa3"].values
        
        
        for i in range(batch_size):
            aux=[]
            for token in q_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>qmax):
                aux=aux[0:qmax]
                len_q[i]=qmax
            q_input[i,0:len(aux)]=aux
            
            aux=[]
            for token in a1_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>amax):
                aux=aux[0:amax]
                len_a1[i]=amax
            a1_input[i,0:len(aux)]=aux
                    
            aux=[]
            for token in wa1_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>amax):
                aux=aux[0:amax]
                len_wa1[i]=amax
            wa1_input[i,0:len(aux)]=aux
                    
            '''
            aux=[self.word_to_index[token] for token in q_aux[i]]
            if(len(aux)>qmax):
                aux=aux[0:qmax]
            q_input[i,0:len(aux)]=aux
        
            aux=[self.word_to_index[token] for token in a1_aux[i]]
            if(len(aux)>amax):
                aux=aux[0:amax]
            a1_input[i,0:len(aux)]=aux
        
            aux=[self.word_to_index[token] for token in wa1_aux[i]]
            if(len(aux)>amax):
                aux=aux[0:amax]
            wa1_input[i,0:len(aux)]=aux
            '''
    
        return q_input, a1_input, wa1_input, len_q, len_a1, len_wa1, df_batch
    
    def get_data_test(self, qmax, amax, df):
        q_input = np.ones((2*len(df), qmax), dtype=int)
        len_q = np.concatenate((df["len_q1"].values,df["len_q1"].values))
        a1_input = np.ones((2*len(df), amax), dtype=int)
        a2_input = np.ones((2*len(df), amax), dtype=int)
        len_a1 = np.concatenate((df["len_a"].values,df["len_c"].values))
        len_a2 = np.concatenate((df["len_b"].values,df["len_d"].values))
        
        q_aux=df["q1"].values        
        a_aux=df["a"].values
        b_aux=df["b"].values
        c_aux=df["c"].values
        d_aux=df["d"].values
        
        for i in range(len(df)):
            #insere os índices das palavras da pergunta em q_input
            aux=[]
            for token in q_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>qmax):
                aux=aux[0:qmax]
                len_q[i]=qmax
                len_q[i+len(df)]=qmax
            q_input[i,0:len(aux)]=aux
            q_input[i+len(df),0:len(aux)]=aux

            #insere os índices das palavras da resposta A em a1_input 
            aux=[]
            for token in a_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>amax):
                aux=aux[0:amax]
                len_a1[i]=amax
            a1_input[i,0:len(aux)]=aux

            #insere os índices das palavras da resposta B em a2_input         
            aux=[]
            for token in b_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>amax):
                aux=aux[0:amax]
                len_a2[i]=amax
            a2_input[i,0:len(aux)]=aux
            #insere os índices das palavras da resposta C em a1_input
            aux=[]
            for token in c_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>amax):
                aux=aux[0:amax]
                len_a1[i+len(df)]=amax
            a1_input[i+len(df),0:len(aux)]=aux
            #insere os índices das palavras da resposta D em a2_input  
            aux=[]
            for token in d_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>amax):
                aux=aux[0:amax]
                len_a2[i+len(df)]=amax
            a2_input[i+len(df),0:len(aux)]=aux
    
        return q_input, a1_input, a2_input, len_q, len_a1, len_a2

    def get_data_cost(self, qmax, amax, df):
        q_input = np.ones((3*len(df), qmax), dtype=int)
        len_q = np.concatenate((df["len_q1"].values,df["len_q1"].values,df["len_q1"].values))
        a1_input = np.ones((3*len(df), amax), dtype=int)
        a2_input = np.ones((3*len(df), amax), dtype=int)
        len_a1 = np.concatenate((df["len_a1"].values,df["len_a1"].values,df["len_a1"].values))
        len_a2 = np.concatenate((df["len_wa1"].values,df["len_wa2"].values,df["len_wa3"].values))
        
        q_aux=df["q1"].values        
        a1_aux=df["a1"].values
        wa1_aux=df["wa1"].values
        wa2_aux=df["wa2"].values
        wa3_aux=df["wa3"].values
        
        for i in range(len(df)):
            #insere os índices das palavras da pergunta em q_input
            aux=[]
            for token in q_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>qmax):
                aux=aux[0:qmax]
                len_q[i]=qmax
                len_q[i+len(df)]=qmax
                len_q[i+2*len(df)]=qmax
            q_input[i,0:len(aux)]=aux
            q_input[i+len(df),0:len(aux)]=aux
            q_input[i+2*len(df),0:len(aux)]=aux

            #insere os índices das palavras contidas na resposta certa em a1_input 
            aux=[]
            for token in a1_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>amax):
                aux=aux[0:amax]
                len_a1[i]=amax
                len_a1[i+len(df)]=amax
                len_a1[i+2*len(df)]=amax
            a1_input[i,0:len(aux)]=aux
            a1_input[i+len(df),0:len(aux)]=aux
            a1_input[i+2*len(df),0:len(aux)]=aux

            #insere os índices das palavras contidas na reposta errada wa1 em a2_input         
            aux=[]
            for token in wa1_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>amax):
                aux=aux[0:amax]
                len_a2[i]=amax
            a2_input[i,0:len(aux)]=aux
            #insere os índices das palavras contidas na reposta errada wa2 em a2_input         
            aux=[]
            for token in wa2_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>amax):
                aux=aux[0:amax]
                len_a2[i+len(df)]=amax
            a2_input[i+len(df),0:len(aux)]=aux
            #insere os índices das palavras contidas na reposta errada wa3 em a2_input         
            aux=[]
            for token in wa3_aux[i]:
                if token not in self.word_to_index:
                    aux.append(2)
                else:
                    aux.append(self.word_to_index[token])
            if(len(aux)>amax):
                aux=aux[0:amax]
                len_a2[i+2*len(df)]=amax
            a2_input[i+2*len(df),0:len(aux)]=aux
    
        return q_input, a1_input, a2_input, len_q, len_a1, len_a2