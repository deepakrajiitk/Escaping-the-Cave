#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pexpect


# In[7]:


child = pexpect.spawn("ssh students@172.27.26.188")
child.expect("students@172.27.26.188's password:")
child.sendline("cs641a")

child.expect("Enter your group name: ", timeout=50)
child.sendline("Turing")

child.expect("Enter password: ", timeout=50)
child.sendline("angry")

child.expect(
    "\r\n\r\n\r\nYou have solved 5 levels so far.\r\nLevel you want to start at: ",
    timeout=50,
)
child.sendline("5")

child.expect(".*")
child.sendline("go")

child.expect(".*")
child.sendline("wave")

child.expect(".*")
child.sendline("dive")

child.expect(".*")
child.sendline("go")

child.expect(".*")
child.sendline("read")
# print(child.before)
child.expect(".*")

f = open("plaintext_file.txt", "r")
f1 = open("ciphertext_file.txt", "w")
plain_list = []
for line in f.readlines():
    li = line.split()
    temp = []
    for l in li:
        child.sendline(l)
        # print(child.before)
        s = str(child.before)[48:64]
        # print(s)
        # if s != " ":
        #     f1.write(s)
        #     f1.write(" ")
        temp.append(s)
        child.expect("Slowly, a new text starts*")
        child.sendline("c")
        child.expect("The text in the screen vanishes!")
    plain_list.append(temp)
    # f1.write("\n")
plain_list[0].pop(0)
for i in range(1, len(plain_list)):
    plain_list[i - 1].append(plain_list[i][0])
    plain_list[i].pop(0)
child.sendline("ffffffffffffffmu")
s = str(child.before)[48:64]
plain_list[len(plain_list) - 1].append(s)

for x in plain_list:
    for y in x:
        f1.write(y)
        f1.write(" ")
    f1.write("\n")
child.close()
f.close()
f1.close()


# In[ ]:
