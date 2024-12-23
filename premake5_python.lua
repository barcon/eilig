-- premake5.lua
workspace "eilig_python"
	configurations { "Python"}
	location "build"

project "eilig_python"
	kind "SharedLib"
	language "C++"
	cppdialect "C++17"
	architecture "x86_64"	
	targetname ("_eilig")
	targetextension (".pyd")

	targetdir "build/%{cfg.buildcfg}"
	includedirs { "../utils/src" }
	includedirs { "../logger/src" }
	
	files "src/eilig.hpp"	
	files "src/eilig_matrix.hpp"	
	files "src/eilig_matrix.cpp"	
	files "src/eilig_matrix_ellpack.hpp"	
	files "src/eilig_matrix_ellpack.cpp"	
	files "src/eilig_routines.hpp"	
	files "src/eilig_routines.cpp"	
	files "src/eilig_status.hpp"	
	files "src/eilig_transform.hpp"	
	files "src/eilig_transform.cpp"	
	files "src/eilig_types.hpp"	
	files "src/eilig_vector.hpp"	
	files "src/eilig_vector.cpp"	

	filter "configurations:Python"
		kind "SharedLib"
		defines { "NDEBUG" }
		optimize "Speed"
		
		includedirs { "../python/inc" }
		
		files "src/eilig_export_python.cpp"	
		
		links { "utils", "logger", "python312" }
		
		libdirs { "../utils/build/Release" }
		libdirs { "../logger/build/Release" }
		libdirs { "../python/lib" }