---
title: "How to Build your Website Using Hugo on GitHub Pages: Summary "
date: 2024-02-01T10:04:47+08:00
draft: false
---
Hugo is a fast and modern static site generator that makes building website much easier. With Hugo, you can host your beautiful website on GitHub Pages, Netlify or other platforms.
## Create a Hugo blog
Before we start, please make sure your hugo is installed. You can check your hugo version by typing 
```
hugo version
```
at your terminal. Next, go to the directory that you want to store website 
```
cd Home
```
for example, and run
```
hugo new site testhugo
```
This command fetches a site template but no content or theme yet. Next, choose your favorite theme and install it. In my case, I choose [PaperMod](https://github.com/adityatelange/hugo-PaperMod) as my theme. For PaperMod, you can install by
```
cd testhugo
git clone https://github.com/adityatelange/hugo-PaperMod themes/PaperMod --depth=1
```
and set theme in `hugo.yaml` (or `.homl` similarly but make sure your syntax is correct)
```
theme: PaperMod
```
To check your website, type 
```
hugo server
```
at your terminal, and you should be able to see
```
Web Server is available at http://localhost:1313/ (bind address 127.0.0.1)
```
It is a local server that dynamically updating your changes on local device. If you want to check how your website looks like, you can turn on `hugo server` and editor at the same time. **However, if your website does not show, you can erase** `baseurl` in your `hugo.yaml` **by setting**
```
baseurl: ""
```
At least for me, I can see my website with this trick. To add a new post, you can
1. type 
```
hugo new content folder-you-want-to-store/first-post.md
```
at your terminal, or

2. simply add it with your editor.

If you use command line to create your post, then you should see
```
+++
title = "first-post"
date = XXXXXX
draft = true
+++
```
at the top of your file. This is a draft post because the variable `draft = true`. If you don't want it to be a draft, you may set `draft = false`. Check it with `hugo server`. If your hugo file is `.yaml` and the post does not show even you set `draft = false`, you can try this change (for format `.yaml`)
```
---
title: "first-post"
date: XXXXXX
draft: true
---
```

## Custom layout
If you want to customize some `.html` files, say `/themes/PaperMod/layouts/partials/footer.html`, here are the steps:
1. Create a folder `partials` with exactly same name under `testhugo/layouts` and copy `footer.html` to `layouts/partials/footer.html`.

2. Customize `footer.html`.

Hugo will overwrite `/themes/PaperMod/layouts/partials/footer.html` with `layouts/partials/footer.html` automatically. Remember, theme settings depends on the theme you choose. Please read the documentation before setting your theme.

## Push to GitHub
Since you have a submodule in your directory, you need to create a file called `.gitmodules` and paste the code into it
```
[submodule "themes/PaperMod"]
    path = themes/PaperMod
    url = https://github.com/adityatelange/hugo-PaperMod.git
```
Go to your website directory and initialize it
```
git init
```

In order to deploy your website, create a new folder `.github/workflows/hugo.yaml` and paste the code (according to the official [documentation](https://gohugo.io/hosting-and-deployment/hosting-on-github/))

```
# Sample workflow for building and deploying a Hugo site to GitHub Pages
name: Deploy Hugo site to Pages

on:
  # Runs on pushes targeting the default branch
  push:
    branches:
      - main

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

# Allow only one concurrent deployment, skipping runs queued between the run in-progress and latest queued.
# However, do NOT cancel in-progress runs as we want to allow these production deployments to complete.
concurrency:
  group: "pages"
  cancel-in-progress: false

# Default to bash
defaults:
  run:
    shell: bash

jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    env:
      HUGO_VERSION: 0.122.0
    steps:
      - name: Install Hugo CLI
        run: |
          wget -O ${{ runner.temp }}/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb \
          && sudo dpkg -i ${{ runner.temp }}/hugo.deb          
      - name: Install Dart Sass
        run: sudo snap install dart-sass
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: recursive
          fetch-depth: 0
      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v4
      - name: Install Node.js dependencies
        run: "[[ -f package-lock.json || -f npm-shrinkwrap.json ]] && npm ci || true"
      - name: Build with Hugo
        env:
          # For maximum backward compatibility with Hugo modules
          HUGO_ENVIRONMENT: production
          HUGO_ENV: production
        run: |
          hugo \
            --gc \
            --minify \
            --baseURL "${{ steps.pages.outputs.base_url }}/"          
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2
        with:
          path: ./public

  # Deployment job
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v3
```
Note that it's `workflows`, not `workflow`. Then commit this change like
```
git add .github/workflows/hugo.yaml
git commit -m "deploy"
```
Now we can push our website to github.
1. Create a new github repository. Don't need to append `.github.io` at the end.
2. Push it to the new repository:
```
git branch -M main           # If you clone from master branch
git remote add https:XXXXXXXX
git push -u origin main
```
If you don't push successfully, then you need to check out the problem. For me, I was asked to input
```
Username for 'https://github.com': 
Password for 'https://your-user-name@github.com':
```
As a newbie, I gave it my "log in" password and the `git push` failed
```
fatal authentication failed ...
```
The password does not mean "log-in" password; instead, it means "personal access tokens". If you don't have one, generate it. If you forget your personal access tokens, please go to the [documentation](https://docs.github.com/en/organizations/managing-programmatic-access-to-your-organization/reviewing-and-revoking-personal-access-tokens-in-your-organization).

Once you successfully push your code, you can go to "Actions" and check your website once the workflow finishes.
## Add images to your post
Suppose you want to insert an image `cat.png` into `first-post.md`. Be careful that you have to
1. Add `cat.png` to the `static` folder. For example, 
```
/static/first-post/cat.png
```
2. Add `baseurl` in front of the path to image!
```
![cat](https://user.github.io/testhugo/first-post/cat.png)
```
Done!
## MathJax setting
1. Paste
```
markup:
  goldmark:
    extensions:
      passthrough:
        delimiters:
          block:
          - - \\(
            - \\)
          - - $$
            - $$
          inline:
          - - \(
            - \)
          - - $
            - $
        enable: true
params:
  math: true
```
in `hugo.yaml`. 

2. Create `mathjax.html` in `/layouts/partials/` to overwrite the theme layout
```
<script>
    MathJax = {
      loader: {
        load: ['[tex]/autoload']
      },
      tex: {
        inlineMath: [ ['$', '$'], ['\(', '\)'] ],
        displayMath: [['$$','$$'], ['\\(', '\\)']],
        packages: {'[+]': ['autoload']}
      },
      svg: {
        fontCache: 'global'
      }
    };
</script>
<script
    type="text/javascript" id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>
```
This means you can activate `inlineMath` by `$math stuff$` or `\(math stuff\)`. The `loader` block manages MathJax extensions. Please refer to [MathJax documentation](https://docs.mathjax.org/en/latest/input/tex/extensions/index.html).

3. Copy `baseof.html` in `theme/PaperMod/layouts/_default/` to 
```
/layouts/_default/baseof.html
```
and add
```
{{ partial "mathjax.html" . }}
```
before `</head>`.

4. Copy `footer.html` in `theme/PaperMod/layouts/partials/` to
```
/layouts/partials/footer.html
```
and append
```
<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
```
at the bottom to enable MathJax. That's all.

### The bug I encounter
I found that `/boldsymbol` does work for me even I added
```
load: ['[tex]/boldsymbol']
```
in `loader` block and 
```
packages: {'[+]': ['boldsymbol']}
```
in `tex` block. If you know how to fix it, please feel free to let me know.

## Put jupyter notebook on your website
We only need to convert `.ipynb` to `.md` using [nbconvert](https://nbconvert.readthedocs.io/en/latest/?badge=latest). Once you install it, run the command line in the directory you want to convert notebooks
```
jupyter nbconvert --to markdown notebook.ipynb
```
Put the converted `.md` file in the `/content/posts.` folder. You may face [image rendering problem](#add-images-to-your-post) or [MathJax](#mathjax-setting) problems. Be patient and don't give up. I put the references so you may have easier time to search. 
# References
- Installation
    - Install Hugo: https://gohugo.io/installation/
    - Install PaperMod: https://github.com/adityatelange/hugo-PaperMod
    - Hugo quickstart: https://gohugo.io/getting-started/quick-start/
    - Host on GitHub Pages: https://gohugo.io/hosting-and-deployment/hosting-on-github/
    - nbconvert installation: https://nbconvert.readthedocs.io/en/latest/install.html
    - Helpful YouTube video: Create A Blog WIth Hugo And Deploy To Github In 10 Minutes, [link](https://www.youtube.com/watch?v=psyz4UPnGAA)
    - Helpful YouTube video: Getting Started With Hugo | FREE COURSE, [link](https://www.youtube.com/watch?v=hjD9jTi_DQ4&t=2299s)
- Debugging:
    - Problems `fatal authentication` for git push:
        - GitHub documentation: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
        - Find your personal access tokens: https://docs.github.com/en/organizations/managing-programmatic-access-to-your-organization/reviewing-and-revoking-personal-access-tokens-in-your-organization
        - Helpful YouTube video: How to fix 'fatal authentication failed for git push' permanently, [link](https://www.youtube.com/watch?v=pHaZW9OWUAQ)
    - MathJax
        - Use MathJax on Hugo: https://hommalab.io/posts/web/use-mathjax-on-hugo/
        - https://bwaycer.github.io/hugo_tutorial.hugo/tutorials/mathjax/
        - https://geoffruddock.com/math-typesetting-in-hugo/
        - MathJax documentation: https://docs.mathjax.org/en/latest/web/configuration.html#configuring-mathjax

### Device
ubuntu, linux, 22.04