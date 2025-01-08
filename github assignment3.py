#!/usr/bin/env python
# coding: utf-8

# # ASSIGNMENT-3

# In[1]:


sub1 = int(input("Enter marks of first subject: "))
sub2 = int(input("Enter marks of second subject: "))
sub3 = int(input("Enter marks of third subject: "))

avg = (sub1+sub2+sub3)/3

if avg>=90:
    print("Grade A")
elif avg>=80 and avg<=89:
    print("Grade B")
elif avg>=70 and avg<=79:
    print("Grade c")
else:
    print("Grade Fail")


# In[ ]:




