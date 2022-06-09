---
# Mr. Green Jekyll Theme - v1.0.1 (https://github.com/MrGreensWorkshop/MrGreen-JekyllTheme)
# Copyright (c) 2022 Mr. Green's Workshop https://www.MrGreensWorkshop.com
# Licensed under MIT

layout: default
---
<div class="multipurpose-container">
  <h1>{{ page.page_header }}</h1>
  <p>{{ page.description }}</p>
  <div class="row"><hr></div>


  {%- assign root_md = site.pages | where_exp: "item", "item.dir == '/notes/'" -%}
  {%- for note_page in root_md -%}     
    <div class="row">
      <div class="col-md-12" style="text-align: right;"><a href="{{ site.baseurl }}{{ note_page.url }}">Go To Page [{{ note_page.title }}]</a></div>
      <div class="col-md-12">{{ note_page.content | markdownify }}</div>
    </div>
    <div class="row"><hr></div>
  {%- endfor -%}

</div>