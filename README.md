# Hill-RSA-Image-Encryption
### Algorithm for Encrypting images using Hill Cipher and RSA :
1. Original image is converted into suitable form. In this proposal we converted the original message image into 256x256 matrix 
2. The 256x256 image is divided into 2x2 sub-matrices.  
3. The 2x2 key matrix is generated. 
4. Performing arithmetic operations using key matrix on the sub-matrix 
5. Convert the resultant image into RGB format and encrypt using advance Hill cipher algorithm using the key matrix 
6. By the side of receiver end, decrypt the image by using the same key matrix which is used in the encryption process
