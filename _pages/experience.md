---
layout: page
title: experience
permalink: /experience/
description: I have been fortunate to work at many great companies with many talented people
nav: true
---

<div class="experiences">

{% assign sorted_exp = site.experiences | sort: "importance" %}
{% for exp in sorted_exp %}
<ol class="explist">
    <li>
        <h4 class="year">{{exp.dates}}</h4>
        <img src="{{ exp.img | relative_url }}" alt="project thumbnail">
        <div class="expinfo">
          <a href="{{ exp.url | relative_url }}">
            <div class="exptitle">{{exp.title}}</div>
            <div class="expdesc">{{exp.position_name}}</div>
          </a>
        </div>
    </li>
</ol>
{% endfor %}

</div>