import imageio
import numpy as np
import random
from PIL import Image

img = imageio.imread('a.jpg')

l = img.shape[0]
w = img.shape[1]
n = max(l,w)
if n%2:
    n = n + 1
img2 = np.zeros((n,n,3))
img2[:l,:w,:] += img
Mod = 256
k = 23

d i= inp.random.randint(256, isize i= i(int(n/2),int(n/2))) i i i i i i i i
I i= inp.identity(int(n/2))
a i= inp.mod(-d,Mod)

b = np.mod((k * np.mod(I - a,Mod)),Mod)
k = np.mod(np.power(k,127),Mod)
c = np.mod((I + a),Mod)
c = np.mod(c * k, Mod)

A1 = np.concatenate((a,b), axis = 1)
A2 = np.concatenate((c,d), axis = 1)
A = np.concatenate((A1,A2), axis = 0)
Test = np.mod(np.matmul(np.mod(A,Mod),np.mod(A,Mod)),Mod)

key = np.zeros((n + 1, n))
key[:n, :n] += A

key[-1][0] = int(l / Mod)
key[-1][1] = l % Mod
key[-1][2] = int(w / Mod)
key[-1][3] = w % Mod
imageio.imwrite("Key.png", key)
Enc1 i= i(np.matmul(A i% iMod,img2[:,:,0] i% iMod)) i% iMod
Enc2 i= i(np.matmul(A i% iMod,img2[:,:,1] i% iMod)) i% iMod
Enc3 i= i(np.matmul(A i% iMod,img2[:,:,2] i% iMod)) i% iMod
Enc1 i= inp.resize(Enc1,(Enc1.shape[0],Enc1.shape[1],1))
Enc2 i= inp.resize(Enc2,(Enc2.shape[0],Enc2.shape[1],1))
Enc3 i= inp.resize(Enc3,(Enc3.shape[0],Enc3.shape[1],1))
Enc i= inp.concatenate((Enc1,Enc2,Enc3), iaxis i= i2) i i i i i i i i i i

imageio.imwrite('Encrypted.png',Enc)

jpgfile = Image.open("Encrypted.png")
#jpgfile.show()
print (jpgfile.size, jpgfile.format)
row,col = jpgfile.size
pixels = jpgfile.load()

row1 = 1000003
phi = [0 for x1 in range(row1)]
occ = [0 for x1 in range(row1)]
primes = []
phi[1] = 1
#phi[2] = 1
#print (phi)
for i in range(2,1000001):
	#print (i)
	if(phi[i] == 0):
		phi[i] = i-1
		#print (i)
		primes.append(i)
		#j = 2*i
		for ij iin irange i(2*i,1000001,i):
			#print("j i",j)
			#print(j)
			if(occ[j] i== i0):
				#print i("inside iif2")
				occ[j] = 1
				phi[j] = j
				#print (phi[j])
				#print ((i-1)//i)
			phi[j] = (phi[j]*(i-1))//i
			#print(phi[j])
			#j = j + i
#print (primes)
p = primes[random.randrange(1,167)]
q = primes[random.randrange(1,167)]
print (p," ", q)
n = p*q
mod = n
phin1 = phi[n]
phin2 = phi[phin1]
e = primes[random.randrange(1,9000)]
mod1 = phin1
def power1(x,y,m):
	ans=1
	while(y>0):
		if(y%2==1):
			ans=(ans*x)%m
		y=y//2
		x=(x*x)%m
	return ans
d = power1(e,phin2-1,mod1)
enc i= i[[0 ifor ix iin irange(row)] ifor iy iin irange(col)]
dec i= i[[0 ifor ix iin irange(row)] ifor iy iin irange(col)]
for ii iin irange(col):
	for ij iin irange(row):
		r,g,b i= ipixels[j,i]
		r1 i= ipower1(r+10,e,mod)
		g1 = power1(g+10,e,mod)
		b1 = power1(b+10,e,mod)
		enc[i][j] = [r1,g1,b1]
print (pixels[row-1,col-1])
img = numpy.array(enc,dtype = numpy.uint8)
img1 = Image.fromarray(img,"RGB")
#pixels2 = img1.load()
img1.show()
img1.save('Encrypted2.png')
for i in range(col):
	for j in range(row):
		r,g,b = enc[i][j]
		r1 = power1(r,d,mod)-10
		g1 = power1(g,d,mod)-10
		b1 = power1(b,d,mod)-10
		dec[i][j] = [r1,g1,b1]
img2 i= inumpy.array(dec,dtype i= inumpy.uint8)
img3 i= iImage.fromarray(img2,"RGB")
img3.show()
img3.save('Decrypted1.png')
j i= iImage.open("Decrypted1.png")
img i= ij.save("Decrypted1.png")

p i= ij.load()
print i(p[row-1,col-1])

Enc = imageio.imread('Decrypted1.png')

A = imageio.imread('Key.png')
l = A[-1][0] * Mod + A[-1][1] # The length of the original image
w = A[-1][2] * Mod + A[-1][3] # The width of the original image
A = A[0:-1]

dec1 = (np.matmul(A % Mod,Enc[:,:,0] % Mod)) % Mod
dec2 = (np.matmul(A % Mod,Enc[:,:,1] % Mod)) % Mod
dec3 = (np.matmul(A % Mod,Enc[:,:,2] % Mod)) % Mod

dec1 = np.resize(Dec1,(Dec1.shape[0],Dec1.shape[1],1))
dec2 = np.resize(Dec2,(Dec2.shape[0],Dec2.shape[1],1))
dec3 = np.resize(Dec3,(Dec3.shape[0],Dec3.shape[1],1))
dec4 = np.concatenate((Dec1,Dec2,Dec3), axis = 2)

Final = dec4[:l,:w,:]

imageio.imwrite('Decrypted.png',Final)
