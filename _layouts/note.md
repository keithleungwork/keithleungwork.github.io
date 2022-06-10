---
# Mr. Green Jekyll Theme - v1.0.1 (https://github.com/MrGreensWorkshop/MrGreen-JekyllTheme)
# Copyright (c) 2022 Mr. Green's Workshop https://www.MrGreensWorkshop.com
# Licensed under MIT

layout: default
---
{% assign crumbs = page.url | remove:'/index.html' | split: '/' %}
<div class="row">
  <a href="/">Home</a>
  {% for crumb in crumbs offset: 1 %}
    {% assign crum_split = crumb | url_decode | remove: '.html' | split: ' ' %}
    {% if crum_split.last.size == 32  %}
      {% assign len = crum_split.size | plus: -1 %}
      {% assign crum_split = crum_split | slice: 0, len  %}
    {% endif %}
    {% assign new_name = crum_split | join: ' ' | replace:'-',' ' | capitalize %}


    {% if forloop.last %}
      / {{ new_name }}
    {% else %}
      / <a href="{% assign crumb_limit = forloop.index %}{% for crumb in crumbs limit: crumb_limit offset: 1 %}{{ crumb | prepend: '/' }}{% endfor %}.html">{{ new_name }}</a>
    {% endif %}
  {% endfor %}
</div>
<br>

<div class="multipurpose-container post-container">
  <!-- <div class="post-title">Note : {{ page.title }}</div> -->
  <!-- <hr/> -->
  <div class="markdown-style">
    {{ content }}
  </div>
</div>
