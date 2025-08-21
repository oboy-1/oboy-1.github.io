---
layout: page
title: All News
permalink: /news
---

<div class="news-timeline">
  <ul class="news-list">
    {% for item in site.data.news %}
    <li class="news-item">
      <div class="news-date">{{ item.date }}</div>
      <div class="news-content">{{ item.content | markdownify | strip_newlines }}</div>
    </li>
    {% endfor %}
  </ul>
</div>

<style>
.news-timeline {
  max-width: 800px;
  margin: 2em auto;
  padding: 0 1em;
  position: relative;
}

.news-list {
  list-style: none;
  padding: 0;
  margin: 0;
  position: relative;
}

/* vertical line */
.news-list::before {
  content: "";
  position: absolute;
  left: 120px; /* adjust for date width */
  top: 0;
  bottom: 0;
  width: 2px;
  background: #ff6600;
  margin-left: -1px;
}

/* each item */
.news-item {
  display: flex;
  align-items: flex-start;
  margin-bottom: 2em;
  position: relative;
}

/* date on the left */
.news-date {
  width: 100px;
  text-align: right;
  padding-right: 20px;
  font-weight: bold;
  font-size: 0.9em;
  flex-shrink: 0;
  color: #000; /* normal color in light mode */
}

/* content box on right */
.news-content {
  background: #fff;
  padding: 1em 1.2em;
  border-radius: 10px;
  border: 1px solid #e0e0e0;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  flex-grow: 1;
  line-height: 1.5em;
  color: #333;
  transition: transform 0.2s, box-shadow 0.2s;
}

.news-content:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* dark mode */
@media (prefers-color-scheme: dark) {
  .news-list::before {
    background: #ff6600;
  }

  .news-date {
    color: #ddd; /* normal color in dark mode */
  }

  .news-content {
    background: #262626;
    border: 1px solid #444;
    color: #ddd;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
  }

  .news-content:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }
}
</style>

