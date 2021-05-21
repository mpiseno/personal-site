---
layout: page
title: experience
permalink: /experience/
description:
nav: true
---

<div class="experiences">

{% assign sorted_exp = site.experiences | sort: "importance" %}
{% for exp in sorted_exp %}
<a href="{{ exp.url | relative_url }}">
    <ol class="explist">
        <li>
            <h4 class="year">{{exp.dates}}</h4>
            <img src="{{ exp.img | relative_url }}" alt="project thumbnail">
            <div class="expinfo">    
                <div class="exptitle">{{exp.title}}</div>
                <div class="expdesc">{{exp.position_name}}</div>
            </div>
        </li>
    </ol>
</a>
{% endfor %}

</div>