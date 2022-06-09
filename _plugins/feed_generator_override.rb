# frozen_string_literal: true

module JekyllFeed
    class Generator < Jekyll::Generator
      safe true
      priority :lowest
  
      # Main plugin action, called by Jekyll-core
      def generate(_site)
        if site.config["feed"]["disable"]
            Jekyll.logger.info "Jekyll Feed disabled"
        else
            old_generate(site)
        end
      end
    end
  end