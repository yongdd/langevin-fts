-- Permit code identifiers/paths to wrap in narrow provenance-table columns.
function Code(el)
  if FORMAT:match('latex') and not el.text:match('[{}\\\n]') then
    local words = {}
    for word in el.text:gmatch('%S+') do
      table.insert(words, '\\nolinkurl{' .. word .. '}')
    end
    return pandoc.RawInline('latex', table.concat(words, '\\allowbreak\\ '))
  end
end
-- Keep a chapter's figures before the next top-level topic.
function Header(el)
  if FORMAT:match('latex') and el.level == 1 then
    return {pandoc.RawBlock('latex', '\\FloatBarrier'), el}
  end
end
-- Use absolute assets because Tectonic compiles inside a temporary directory.
function Image(el)
  if not el.src:match('^/') and not el.src:match('^https?://') then
    el.src = os.getenv('CHARGED_REPORT_ROOT') .. '/' .. el.src
  end
  return el
end
