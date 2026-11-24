-- PDF build only: use the vector PDF that sits next to each PNG figure (GitHub needs the PNGs, LaTeX prefers PDFs).
function Image(img)
  if img.src:match("%.png$") then
    local pdf = img.src:gsub("%.png$", ".pdf")
    local f = io.open(pdf, "r")
    if f then
      f:close()
      img.src = pdf
    end
  end
  return img
end
