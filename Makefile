default:
	echo "hi"

start: replace_link
	bundle exec jekyll serve --watch --safe

# To fix the encoded links in md
replace_link:
	node ./notion2md/search_replace.js


build: replace_link
	bundle exec jekyll build --profile
