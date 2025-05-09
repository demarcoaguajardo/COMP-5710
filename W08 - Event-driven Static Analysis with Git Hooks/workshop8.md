## Workshop 8

## Workshop Name: Event-driven Static Analysis with Git Hooks

## Description 

Use an existing tool and Git Hooks to activate a static analysis tool for a popular repository 

## Targeted Courses 

Software Quality Assurance 

## Activities 

### Pre-lab Content Dissemination 

One negative perception about software quality assurance (SQA) is that it prohibits rapid deployment of software. That is why practitioners advocate SQA activities to be integrated into the software development and deployment process. To that end, in modern software engineering, practitioners prefer automated pipelines for security analysis. Instead of asking practitioners to look for security problems themselves, tools should do that for them. 

In that spirit, we as a class  will build a mini tool that automatically runs static security analysis for [FastJM](https://github.com/shanpengli/FastJM), a popular library developed in C/C++. For this workshop you will use [cppcheck](https://cppcheck.sourceforge.io/) and `git hooks`. You will build a Git Hook that will help in identifying known security weaknesses automatically for practitioners who develop and use `FastJM`.  


### In-class Hands-on Experience 

- Create a GitHub account if you haven't yet 
- Install CPPCheck on your computer 
- Clone the `FastJM` repository on your computer  
- Go to `.git/hooks/` in the cloned repository 
- Run `cp pre-commit.sample pre-commit` 
- Open `pre-commit` 
- Edit `pre-commit` to run `cppcheck -h`
- Familiarize yourself with `cppcheck` using any or all of the following links: 
   - https://linux.die.net/man/1/cppcheck 
   - http://cppcheck.sourceforge.net/manual.pdf 
   - https://www.mankier.com/1/cppcheck 
- Modify any `.c` or `.cpp` file 
- Commit the modified file to see the effects of the modified `pre-commit` hook   
- Recording of this hands-on experience is available on CANVAS 

### Post Lab Experience
- Modify your `pre-commit` file so that it can scan your [NumCPP](https://github.com/dpilger26/NumCpp) repository whenever you commit any file
  - Grab your output by capturing the screenshots 
  - Modify any CPP file in the `NumCPP` repository 
  - Upload your `pre-commit` file and your screenshots on CANVAS 