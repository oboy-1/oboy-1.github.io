#!/usr/bin/env ruby
#
# Check for changed posts

Jekyll::Hooks.register :posts, :post_init do |post|

  commit_num = `git rev-list --count HEAD "#{ post.path }"`

  if commit_num.to_i > 1
    # Commits tagged "[skip-lastmod]" (e.g. formatting-only refactors that
    # don't change what the post says) are ignored here, so the displayed
    # "last modified" date only reflects substantive edits.
    lastmod_date = `git log -1 --invert-grep --grep="\\[skip-lastmod\\]" --pretty="%ad" --date=iso "#{ post.path }"`
    # Fall back to the plain history if every commit happens to be tagged.
    lastmod_date = `git log -1 --pretty="%ad" --date=iso "#{ post.path }"` if lastmod_date.strip.empty?
    post.data['last_modified_at'] = lastmod_date
  end

end
