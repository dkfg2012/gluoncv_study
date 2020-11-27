from mxnet import nd

matrix = nd.array(((1,2,3),(5,6,7))) #create matrix
ones_matrix = nd.ones((2,3)) # create 2*3 matrix contain all 1
random_matrix = nd.random.uniform(-1, 1, (2,3)) #create 2*3 matrix with element is a random between -1 and 1
two_matrix = nd.full((2,3),2.0) #create 2*3 matrix with all element equal to 2

#indexing
random_matrix[1,2] #return element locate index 1 row and index 2 column
random_matrix[:,2] #return column vector index at 2
random_matrix[:,1:3] #return matrix contain column vector index at 1 and 2

random_matrix = random_matrix.asnumpy() #convert the matrix into numpy format
random_matrix = nd.array(random_matrix) #convert numpy matrix into nd matrix format