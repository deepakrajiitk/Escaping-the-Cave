#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:
chars="fghijklmnopqrstu"
char_dict={}
for j in range(16):
    char_dict[str(np.binary_repr(j, width=4))]=chars[j]
# print(char_dict)

# In[2]:


file = open("plaintext_file.txt","w+")

for i in range(8):
    for j in range(128):
        binary = np.binary_repr(j, width=8)
        plaintext = 'ff'*i + char_dict[binary[:4]] + char_dict[binary[4:]] + 'ff'*(8-i-1)
        file.write(plaintext)
        file.write(" ")
    file.write("\n")
file.close()


# %%
