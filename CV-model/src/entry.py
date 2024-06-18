from js import Response
from time import sleep
import pickle
import json
import os 
import math
import json
import http.client


def shape(x):
	return (len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]))


def conv_forward(x,weight,b,parameters):

	pad = parameters['pad']
	stride = parameters['stride']

	(m, n_h, n_w, n_C_prev) = shape(x)
	(f, f, n_C_prev, n_C) = shape(weight)

	n_H = int(1 + (n_h + 2 * pad - f) / stride)
	n_W = int(1 + (n_w + 2 * pad - f) / stride)

	samples, x_width, y_height, channels = shape(x)

	x_prev_pad = [[[[0 for _ in range(channels)] for _ in range(y_height + 2*pad)] for _ in range(x_width + 2*pad)] for _ in range(samples)]

	# Padding operation (expensive but fine when i is small)
	for i in range(samples):
		for j in range(x_width):
			for k in range(y_height):
				for l in range(channels):
					x_prev_pad[i][j + 1][k + 1][l] = x[i][j][k][l]

	Z = [[[[0 for _ in range(n_C)] for _ in range(n_W)] for _ in range(n_H)] for _ in range(m)]

	caches = (x,weight,b,pad,stride)
	
	# Convolution operation (expensive but fine when i is small) 
	for i in range(m):
		for h in range(n_H):
			for w in range(n_W):
				for c in range(n_C):

						vert_start = h*stride
						vert_end = vert_start + f
						horiz_start = w * stride
						horiz_end = horiz_start + f

						for x in range(f):
							for y in range(f):
								for j in range(len(x_prev_pad[0][0][0])):
									Z[i][h][w][c] += x_prev_pad[i][vert_start + x][horiz_start + y][j] * weight[x][y][j][c]

						# Add bias term 
						Z[i][h][w][c] += b[c]
	return Z


def batchnorm(x, running_mu, running_sigma):

	m,h,w,c = shape(x)

	mu = [[[running_mu]]]
	sigma = [[[running_sigma]]]

	for i in range(m):
		for j in range(h):
			for k in range(w):
				for l in range(c):
					x[i][j][k][l] = (x[i][j][k][l] - mu[0][0][0][l])/math.sqrt(sigma[0][0][0][l] + 1e-8)

	return x


def relu(x):

	for i in range(len(x)):
		for j in range(len(x[0])):
			for k in range(len(x[0][0])):
				for l in range(len(x[0][0][0])):
					x[i][j][k][l] = max(0, x[i][j][k][l])
	return x


def max_pooling(prev_layer, filter_size=2):

	(m, n_H_prev, n_W_prev, channels) = shape(prev_layer)

	stride = 2

	# With max pooling I dont want overlapping filters so make stride = filter size
	n_H = int((n_H_prev - filter_size)/filter_size + 1)
	n_W = int((n_W_prev - filter_size)/filter_size + 1)

	pooling = [[[[0 for _ in range(channels)] for _ in range(n_W)] for _ in range(n_H)] for _ in range(m)]

	for i in range(m):
		for h in range(n_H):
			for w in range(n_W):
				for c in range(channels):

					vert_start = h*filter_size
					vert_end = vert_start + filter_size
					horiz_start = w*filter_size
					horiz_end = horiz_start + filter_size
									
					for x in range(filter_size):
						for y in range(filter_size):
							pooling[i][h][w][c] = max(pooling[i][h][w][c], prev_layer[i][vert_start + x][horiz_start + y][c])
		
	return pooling

def fully_connected(prev_layer, w,b):

	output = [[0 for i in range(10)] for j in range(len(prev_layer))]

	for row in range(len(prev_layer)):
		for col in range(10):
			for entry in range(len(prev_layer[0])):
				output[row][col] += prev_layer[row][entry]*w[entry][col]  + b[row]
		
	return output


def softmax(z):

	# Softmax term
	for i in range(len(z)):
		row_sum = 0
		for j in range(len(z[0])):
			row_sum += math.exp(z[i][j])

		for j in range(len(z[0])):
			z[i][j] = math.exp(z[i][j])/ row_sum
			
	return z

def preprocess(x):
	
	# Preprocessing step on the input
	(m, n_h, n_w, n_C_prev) = shape(x)
	
	for i in range(m):
		for h in range(n_h):
			for w in range(n_w):
				for c in range(n_C_prev):
					x[i][h][w][c] = x[i][h][w][c] - 0.43326861213235296

def final_layer_reshape(Pool3):

	(a0, a1, a2, a3) = shape(Pool3)
	pool3_reshape = [[] for _ in range(a0)]

	for i in range(a0):
		for j in range(a1):
			for k in range(a2):
				for l in range(a3):
					pool3_reshape[i].append(Pool3[i][j][k][l]) 

	return pool3_reshape

def evaluate(x, weights):

		preprocess(x)

		# Layer 1
		Z1 = conv_forward(x,weights["W1"],weights["B1"],{'pad':1,'stride':1})
		BN1 = batchnorm(Z1, weights["running_mu_1"], weights["running_sigma_1"])
		A1 = relu(BN1)
		Pool1 = max_pooling(A1,2)
		# Layer 2
		Z2 = conv_forward(Pool1,weights["W2"],weights["B2"],{'pad':1,'stride':1})
		BN2 = batchnorm(Z2, weights["running_mu_2"], weights["running_sigma_2"])
		A2 = relu(BN2)
		Pool2 = max_pooling(A2,2)
		# Layer 3
		Z3 = conv_forward(Pool2,weights["W3"],weights["B3"],{'pad':1,'stride':1})
		BN3 = batchnorm(Z3, weights["running_mu_3"], weights["running_sigma_3"])
		A3 = relu(BN3)
		Pool3 = max_pooling(A3,2)
		pool3_reshape = final_layer_reshape(Pool3)
		# Final Layer 
		Z4 = fully_connected(pool3_reshape, weights["W4"], weights["B4"])

		# Softmax Layer 
		A4 = softmax(Z4)

		return A4


async def on_queue(batch, env, ctx):

	weights = {
		"W1": json.loads(await env.AIMICRO.get("W1")),
		"W2": json.loads(await env.AIMICRO.get("W2")),
		"W3": json.loads(await env.AIMICRO.get("W3")),
		"W4": json.loads(await env.AIMICRO.get("W4")),
		"B1": json.loads(await env.AIMICRO.get("B1")),
		"B2": json.loads(await env.AIMICRO.get("B2")),
		"B3": json.loads(await env.AIMICRO.get("B3")),
		"B4": json.loads(await env.AIMICRO.get("B4")),
		"running_mu_1": json.loads(await env.AIMICRO.get("running_mu_1")),
		"running_mu_2":  json.loads(await env.AIMICRO.get("running_mu_2")),
		"running_mu_3": json.loads(await env.AIMICRO.get("running_mu_3")),
		"running_sigma_1": json.loads(await env.AIMICRO.get("running_sigma_1")),
		"running_sigma_2": json.loads(await env.AIMICRO.get("running_sigma_2")),
		"running_sigma_3": json.loads(await env.AIMICRO.get("running_sigma_3"))
	}

	for message in batch.messages:
		image = json.loads(message.body.image)
		await env.AIMICRO.put(str(message.body.UUID), str(evaluate([image], weights)))
		

async def on_fetch(request, env):
	return Response.new("OK")

