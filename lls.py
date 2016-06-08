import numpy as np
import math
import sys
from sklearn import linear_model
from scipy import stats

class lls_impute:
	def __init__(self,dat,mdata):
	   	self.arr = dat
	   	self.shp = self.arr.shape
		self.shp1 = self.shp[0]
		self.shp2 = self.shp[1]
		self.distlist = np.zeros((self.shp1,self.shp1))
		self.missposs = np.copy(mdata)

	def get_missrow(self,n1):
		lisy = []
		for h in range(self.shp2):
			if self.check_missing(n1,h) == 1:
				lisy.append(h)
		return lisy

	def check_missing(self,a,b):
		return self.missposs[a][b]

	def distance2(self,n1,n2):
		if self.distlist[n1][n2]==0 and self.distlist[n2][n1]==0: 			#pearson distance
			distx = stats.pearsonr(self.arr[n1],self.arr[n2])
			distt2 = distx[0]
			self.distlist[n1][n2] = distt2
			self.distlist[n2][n1] = distt2	
			return distt2	
		else:
			return self.distlist[n1][n2]			
		
	def predict_it(self,rowno,atts,listy):			#to predict multiple attributes in a row
		predx = []									#rowno = the row which to predict, #listy = top20 similar rows
		#print 'to predict, rowno:',rowno
		yatts = []
		for j in range(self.shp2):
			if j not in atts:
				yatts.append(j)
		now = self.arr[listy,:]
		datanow_x = now[:,yatts]
		datanow_y = now[:,atts]
		data_test_x = self.arr[rowno,yatts]
		regr = linear_model.LinearRegression()
		regr.fit(datanow_x,datanow_y)
		predicted = regr.predict(data_test_x)
		return predicted

	def distance2weights(self,brr):
		maxim = max(brr)+1
		for j in range(len(brr)):
			brr[j] = maxim-brr[j]
		sumim = sum(brr)
		for k in range(len(brr)):
			brr[k] = brr[k]/float(sumim) 				#except ZeroDivisionError:
		return brr

	def find_top(self,yy,n1):			#non_weighted
		listy = []
		for k in range(self.shp1):
			if k!=n1:
				distx = self.distance2(n1,k)
				listy.append(tuple([k,distx]))
		sorted_by_second = sorted(listy, key=lambda tup: tup[1],reverse=True)
		if len(sorted_by_second)>yy:						#here yy is the k, in top-k
			tempr = sorted_by_second[:yy]
		else:
			tempr = sorted_by_second					#selecting top 10 or top yy
		err = []
		for xx in tempr:
			err.append(xx[0])
		return err

	def write_back(self,finx,fname):
		ofile = open(fname,'w')
		for j in range(self.shp1):
			for k in range(self.shp2):
				ofile.write(str(finx[j][k])+" ")
			ofile.write("\n")
		ofile.close()

	def check(self,brr):
		mism = 0
		for j in range(self.shp1):
			for k in range(self.shp2):
				if self.arr[j][k]!=brr[j][k]:
					mism+=1
		#print "mismatched = ",mism

	def drawProgressBar(self,percent, barLen = 50):
	    sys.stdout.write("\r")
	    progress = ""
	    for i in range(barLen):
	        if i<int(barLen * percent):
	            progress += "="
	        else:
	            progress += " "
	    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
	    sys.stdout.flush()

	def predict_all(self):
		#print 'predict all called!'
		finale = np.copy(self.arr)
		predcount = 0
		for g in range(self.shp1):
			misx = self.get_missrow(g)			#returns the index of the missing values in row g 
			#print '\nget_missrow:', misx
			if len(misx)>0:
				hell = self.find_top(20,g)					#returns the 10 most similar instances
				#print 'hell top20=',hell
				predictedx = self.predict_it(g,misx,hell)
				#print 'predictedx=',predictedx
				for j in range(len(misx)):
					mnx = predictedx[0][j]			#calculates the mean 
					indt = misx[j]
					finale[g][indt]=mnx
				predcount+=len(predictedx)
				pc = g/float(self.shp1)
				self.drawProgressBar(pc)
		self.drawProgressBar(1)
		print "\n"
		return finale
