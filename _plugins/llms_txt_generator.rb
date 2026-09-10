#!/usr/bin/env ruby
#
# Generates /llms.txt (a site-wide guide for LLMs/agents) plus a plain-markdown
# companion file per post (e.g. /posts/how-pca-works.md alongside
# /posts/how-pca-works/), so an agent fetching the site gets a teaching guide
# and clean, iframe-free source text instead of having to scrape rendered HTML.

IFRAME_RE = /<iframe\b[^>]*>.*?<\/iframe>/im.freeze
SRC_ATTR_RE = /\bsrc\s*=\s*"([^"]*)"/i.freeze
TITLE_ATTR_RE = /\btitle\s*=\s*"([^"]*)"/i.freeze

# Capture each post's body markdown (front matter stripped, Liquid resolved,
# pre-layout) right after conversion -- this is the one point in the render
# pipeline where content is clean prose without sidebar/nav chrome.
Jekyll::Hooks.register :posts, :pre_render do |post, payload|
  post.data['llms_raw_source'] = post.content.dup
end

Jekyll::Hooks.register :site, :post_write do |site|
  posts = site.posts.docs.select { |p| p.data['llms_raw_source'] }

  write_post_companions(site, posts)
  write_llms_txt(site, posts)
end

def write_post_companions(site, posts)
  posts.each do |post|
    body = resolve_baseurl(post.data['llms_raw_source'], site)
    body = replace_iframes(body, post, site)

    dest_path = File.join(site.dest, post.url.chomp('/') + '.md')
    FileUtils.mkdir_p(File.dirname(dest_path))

    front = ["# #{post.data['title']}"]
    front << "Date: #{post.date.strftime('%Y-%m-%d')}" if post.date
    tags = Array(post.data['tags'])
    front << "Tags: #{tags.join(', ')}" unless tags.empty?

    File.write(dest_path, "#{front.join("\n")}\n\n#{body.strip}\n")
  end
end

def write_llms_txt(site, posts)
  guide = site.data['llm_guide'] || {}
  lines = []
  lines << "# Karthik's Blog — Guide for LLMs"
  lines << ''
  lines << "> #{guide['intro'].to_s.strip}" if guide['intro']
  lines << ''
  lines << '## How to teach from this site'
  Array(guide['principles']).each { |p| lines << "- #{p}" }
  lines << ''
  lines << '## Posts'

  posts.sort_by { |p| p.date }.reverse_each do |post|
    lines << ''
    lines << "### #{post.data['title']}"
    lines << "- Page: #{absolute_url(site, post.url)}"
    lines << "- Clean content: #{absolute_url(site, post.url.chomp('/') + '.md')}"

    notes = post.data['llm_teaching_notes']
    lines << "- Teaching notes: #{notes.to_s.strip}" if notes

    interactives = Array(post.data['llm_interactives'])
    unless interactives.empty?
      lines << '- Interactives (fetch to embed):'
      interactives.each do |i|
        lines << "  - #{absolute_url(site, i['src'])} — #{i['when']}"
      end
    end
  end

  File.write(File.join(site.dest, 'llms.txt'), "#{lines.join("\n")}\n")
end

def resolve_baseurl(text, site)
  text.gsub(/\{\{\s*site\.baseurl\s*\}\}/, site.config['baseurl'].to_s)
end

def absolute_url(site, path)
  return path if path.to_s.include?('://')

  "#{site.config['url']}#{site.config['baseurl']}#{path}"
end

def replace_iframes(body, post, site)
  interactives = Array(post.data['llm_interactives'])

  body.gsub(IFRAME_RE) do |tag|
    src = tag[SRC_ATTR_RE, 1]
    next tag unless src

    matched = interactives.find { |i| i['src'] == src }
    title = matched && matched['title']
    title ||= tag[TITLE_ATTR_RE, 1]
    title ||= 'Interactive visualization'

    "[Interactive: #{title}](#{absolute_url(site, src)})"
  end
end
