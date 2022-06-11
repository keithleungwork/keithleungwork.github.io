# My blog
[My dear blog](https://keithleungwork.github.io/)

## Outline

- [My blog](#my-blog)
  - [Outline](#outline)
  - [Test locally](#test-locally)
    - [Explain: What replace_link does](#explain-what-replace_link-does)
  - [Configuration](#configuration)
    - [_config.yml](#_configyml)
    - [_data/owner.yml](#_dataowneryml)
  - [Tech Stack](#tech-stack)
  - [Coding Guide](#coding-guide)
    - [Table of content](#table-of-content)

## Test locally

1. Install all prerequisite [here](https://jekyllrb.com/docs/)
2. Run `bundle install`
3. Run `make start`


### Explain: What replace_link does

First of all, most of my markdown is exported directly from Notion.
The links in those exported markdown is encoded while github Pages cannot recognize them well, during the compiling.
So I made the nodejs script `./notion2md/search_replace.js` to decode all the links in markdown files under `notes` directory.




## Configuration

### _config.yml
It is the main configuration of jekyll, for building the site.


### _data/owner.yml
Here is the main configuration of the theme.





## Tech Stack

- Jekyll
- Theme is from [MrGreen-JekyllTheme](https://github.com/MrGreensWorkshop/MrGreen-JekyllTheme)

-----

## Coding Guide

### Table of content

By using the gem `jekyll-toc`, put `{{ content | toc }}` in any layout and it print out the TOC