import re
import os

path = '../'

'quickstart_generator'
with open(path+'docs/quickstart_template.txt') as f:
    template = f.readlines()

    new_file = []
    for l in template:
        if l[:2] != '& ':
            new_file.append(l)
        else:
            try:
                with open('../test/'+l[2:].replace('\n',''), encoding='utf8') as f:
                    lines = f.readlines()

                    for ind,l in enumerate(lines):
                        if l[:18] == "from expy import *":
                            break

                    code = ''.join(lines[ind:])

                new_file.append('\n```python\n%s\n```\n' %(code))
            except:
                    print('cannnot read',file)


    with open(path+'docs/quickstart.md','w+', encoding='utf8') as f:
        f.writelines(new_file)

'cookbook_generator'
with open(path+'docs/cookbook_template.txt') as f:
    template = f.readlines()

    new_file = []
    for l in template:
        if l[:2] != '& ':
            new_file.append(l)
        else:
            try:
                with open('../test/'+l[2:].replace('\n',''), encoding='utf8') as f:
                    lines = f.readlines()

                    for ind,l in enumerate(lines):
                        if l[:18] == "from expy import *":
                            ind += 1
                            break

                    code = ''.join(lines[ind:])

                new_file.append('\n```python\n%s\n```\n' %(code))
            except:
                    print('cannnot read',file)


    with open(path+'docs/cookbook.md','w+', encoding='utf8') as f:
        f.writelines(new_file)

'api_generator'
with open(path+'docs/api_template.txt') as f:
    template = f.readlines()

    comments = dict()
    for root, _dirs, files in os.walk(path):
        for file in files:
            if file[-3:]=='.py' and root[-4:] != 'test':
                try:
                    with open(root+'\\'+file,encoding='utf8') as f:
                        
                        lines = f.readlines()

                        defines = []
                        for ind,l in enumerate(lines):
                            if  l[:4] == "def ":
                                defines.append([l,ind])

                        for define,ind in defines:
                            if lines[ind+1] == "    '''\n":
                                f = re.search('def (.*?)\((.*)\)',define)
                                f_name = f.group(1)
                                f_args = f.group(2)
                                comments[f_name] = [f_args,'']
                                ind += 2
                                while lines[ind] != "    '''\n":
                                    comments[f_name][1] += (lines[ind][4:-1] + '\n')
                                    ind += 1
                except:
                    print('cannnot read',file)

    new_file = []
    for l in template:
        if '- **' != l[:4]:
            new_file.append(l)
        else:
            f_args,comm = comments[l[4:-3]]
            new_file.append('\n- **%s(%s)**\n\n' %(l[4:-3],f_args))
            new_file.append('\n```\n%s```\n\n' %(comm))

    with open(path+'docs/api.md','w+', encoding='utf8') as f:
        f.writelines(new_file)