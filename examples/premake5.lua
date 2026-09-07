-- premake5.lua
defines { "GLOBAL" }
	language "C++"
	cppdialect "C++20"
	architecture "x86_64" 	

	objdir "%{cfg.location}/obj/%{cfg.platform}_%{cfg.buildcfg}"	
	targetdir "build/%{cfg.buildcfg}"
	
workspace "examples"
	configurations { "Release", "ReleaseCL"}
	location "build"	

project "opencl_vector"
	kind "ConsoleApp"
	targetname ("opencl_vector")		
	files { "src/opencl_vector.cpp" }	
	
	filter {"configurations:Release", "action:gmake*", "toolset:gcc" }   
		defines { "NDEBUG" }
		optimize "Speed"

		includedirs { "../../utils/src" }	
		includedirs { "../../logger/src" }	
		includedirs { "../../eilig/src" }	

		libdirs { "../../utils/build/Release" }
		libdirs { "../../logger/build/Release" }
	
		links { "eilig", "logger", "utils" }
		
	filter {"configurations:Release", "action:vs*" }   
		defines { "NDEBUG" }
		optimize "Speed"
	
		includedirs { "../../utils/src" }	
		includedirs { "../../logger/src" }	
		includedirs { "../../eilig/src" }

		libdirs { "../../utils/build/Release" }
		libdirs { "../../logger/build/Release" }
		
		links { "eilig", "logger", "utils" }	
		
	filter {"configurations:ReleaseCL", "action:gmake*", "toolset:gcc" }   
		defines { "NDEBUG", "EILIG_ENABLE_OPENCL" }
		optimize "Speed"
		
		includedirs { "../../utils/src" }	
		includedirs { "../../logger/src" }	
		includedirs { "../../eilig/src" }	
		includedirs { "../../club/src" }	
		includedirs { "../../opencl/inc" }	

		libdirs { "../../utils/build/Release" }
		libdirs { "../../logger/build/Release" }		
		libdirs { "../../eilig/build/ReleaseCL" }
		libdirs { "../../club/build/ReleaseCL" }
		libdirs { "../../opencl/lib/x86_64" }

		links {"opencl", "club", "eilig", "logger", "utils" }
		
	filter {"configurations:ReleaseCL", "action:vs*" }   
		defines { "NDEBUG", "EILIG_ENABLE_OPENCL" }
		optimize "Speed"
		
		includedirs { "../../utils/src" }	
		includedirs { "../../logger/src" }	
		includedirs { "../../eilig/src" }	
		includedirs { "../../club/src" }	
		includedirs { "../../opencl/inc" }	

		libdirs { "../../utils/build/Release" }
		libdirs { "../../logger/build/Release" }		
		libdirs { "../../eilig/build/ReleaseCL" }
		libdirs { "../../club/build/ReleaseCL" }
		libdirs { "../../opencl/lib/x86_64" }

		links {"opencl", "club", "eilig", "logger", "utils" }

project "opencl_matrix"
	kind "ConsoleApp"
	targetname ("opencl_matrix")		
	files { "src/opencl_matrix.cpp" }	
	
	filter {"configurations:Release", "action:gmake*", "toolset:gcc" }   
		defines { "NDEBUG" }
		optimize "Speed"

		includedirs { "../../utils/src" }	
		includedirs { "../../logger/src" }	
		includedirs { "../../eilig/src" }	

		libdirs { "../../utils/build/Release" }
		libdirs { "../../logger/build/Release" }
	
		links { "eilig", "logger", "utils" }
		
	filter {"configurations:Release", "action:vs*" }   
		defines { "NDEBUG" }
		optimize "Speed"
	
		includedirs { "../../utils/src" }	
		includedirs { "../../logger/src" }	
		includedirs { "../../eilig/src" }

		libdirs { "../../utils/build/Release" }
		libdirs { "../../logger/build/Release" }
		
		links { "eilig", "logger", "utils" }	
		
	filter {"configurations:ReleaseCL", "action:gmake*", "toolset:gcc" }   
		defines { "NDEBUG", "EILIG_ENABLE_OPENCL" }
		optimize "Speed"
		
		includedirs { "../../utils/src" }	
		includedirs { "../../logger/src" }	
		includedirs { "../../eilig/src" }	
		includedirs { "../../club/src" }	
		includedirs { "../../opencl/inc" }	

		libdirs { "../../utils/build/Release" }
		libdirs { "../../logger/build/Release" }		
		libdirs { "../../eilig/build/ReleaseCL" }
		libdirs { "../../club/build/ReleaseCL" }
		libdirs { "../../opencl/lib/x86_64" }

		links {"opencl", "club", "eilig", "logger", "utils" }
		
	filter {"configurations:ReleaseCL", "action:vs*" }   
		defines { "NDEBUG", "EILIG_ENABLE_OPENCL" }
		optimize "Speed"
		
		includedirs { "../../utils/src" }	
		includedirs { "../../logger/src" }	
		includedirs { "../../eilig/src" }	
		includedirs { "../../club/src" }	
		includedirs { "../../opencl/inc" }	

		libdirs { "../../utils/build/Release" }
		libdirs { "../../logger/build/Release" }		
		libdirs { "../../eilig/build/ReleaseCL" }
		libdirs { "../../club/build/ReleaseCL" }
		libdirs { "../../opencl/lib/x86_64" }

		links {"opencl", "club", "eilig", "logger", "utils" }