#!/usr/bin/env ruby
#
# Generates /llms.txt (a site-wide guide for LLMs/agents) plus a plain-markdown
# companion file per post (e.g. /posts/how-pca-works.md alongside
# /posts/how-pca-works/), so an agent fetching the site gets a teaching guide
# and clean, iframe-free source text instead of having to scrape rendered HTML.

# Raw <iframe> tags (posts not yet migrated to the interactive.html include).
IFRAME_RE = /<iframe\b[^>]*>.*?<\/iframe>/im.freeze
SRC_ATTR_RE = /\bsrc\s*=\s*"([^"]*)"/i.freeze
TITLE_ATTR_RE = /\btitle\s*=\s*"([^"]*)"/i.freeze

# {% include interactive.html ... %} calls (see _includes/interactive.html).
# Its "alt" param is a hidden, crawler/LLM-only description of the embed --
# exactly the kind of content this generator wants, already sitting right
# next to the embed instead of needing to be duplicated into front matter.
INCLUDE_RE = /\{%-?\s*include\s+interactive\.html\s*(.*?)-?%\}/m.freeze

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
    body = replace_embeds(body, post, site)

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

    interactives = interactives_for(post, site)
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

# Front-matter `llm_interactives` entries (curated `when`/`title` overrides)
# take priority when present; otherwise fall back to what's auto-discovered
# from the post's own {% include interactive.html %} calls.
def interactives_for(post, site)
  curated = Array(post.data['llm_interactives'])
  return curated unless curated.empty?

  discover_includes(post.data['llms_raw_source']).map do |embed|
    { 'src' => embed[:src], 'title' => embed[:title], 'when' => embed[:alt] }
  end
end

def discover_includes(raw_source)
  raw_source.to_s.scan(INCLUDE_RE).map do |(params)|
    src = extract_attr(params, 'src')
    next nil unless src

    { src: src, title: extract_attr(params, 'title'), alt: extract_attr(params, 'alt') }
  end.compact
end

def extract_attr(blob, name)
  match = blob.match(/\b#{name}\s*=\s*"((?:[^"\\]|\\.)*)"/)
  match && unescape_liquid_string(match[1])
end

def unescape_liquid_string(str)
  str.gsub('\\"', '"').gsub('\\\\', '\\')
end

def replace_embeds(body, post, site)
  curated = Array(post.data['llm_interactives'])

  body = body.gsub(INCLUDE_RE) do |tag|
    params = Regexp.last_match(1)
    src = extract_attr(params, 'src')
    next tag unless src

    matched = curated.find { |i| i['src'] == src }
    title = (matched && matched['title']) || extract_attr(params, 'title') || 'Interactive visualization'
    desc = (matched && matched['when']) || extract_attr(params, 'alt')

    link = "[Interactive: #{title}](#{absolute_url(site, src)})"
    desc ? "#{link} — #{desc}" : link
  end

  body.gsub(IFRAME_RE) do |tag|
    src = tag[SRC_ATTR_RE, 1]
    next tag unless src

    matched = curated.find { |i| i['src'] == src }
    title = matched && matched['title']
    title ||= tag[TITLE_ATTR_RE, 1]
    title ||= 'Interactive visualization'

    "[Interactive: #{title}](#{absolute_url(site, src)})"
  end
end
