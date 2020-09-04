from pygost import gost34112012256

variant = gost34112012256.new("Фирфаров Александр Сергеевич".encode('utf-8')).digest().hex()[-1]
print(variant)
